package claw

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/looplj/axonhub/axon/agent"
)

const (
	defaultContextMaxRecentMessages = 120
	defaultContextSoftTokenLimit    = 160_000
)

type ContextManagerConfig struct {
	Enabled           bool
	MaxRecentMessages int
	SoftTokenLimit    int
	Summarizer        *SmartSummarizer
	Logger            *slog.Logger
}

func DefaultContextManagerConfig() ContextManagerConfig {
	return ContextManagerConfig{
		Enabled:           true,
		MaxRecentMessages: defaultContextMaxRecentMessages,
		SoftTokenLimit:    defaultContextSoftTokenLimit,
	}
}

const compactionCooldown = 30 * time.Second

type SmartContextManager struct {
	agent.ContextManager

	config ContextManagerConfig
	store  ContextManagerStore
	logger *slog.Logger

	mu              sync.RWMutex
	state           agent.ContextManagerState
	lastCompactedAt time.Time
	onCompaction    func()
}

func NewSmartContextManager(config ContextManagerConfig, store ContextManagerStore) (*SmartContextManager, error) {
	return NewSmartContextManagerWithNext(nil, config, store)
}

func NewSmartContextManagerWithNext(next agent.ContextManager, config ContextManagerConfig, store ContextManagerStore) (*SmartContextManager, error) {
	cfg := mergeDefaultContextManagerConfig(config)
	if cfg.Summarizer == nil {
		return nil, fmt.Errorf("context manager summarizer is required")
	}

	if next == nil {
		next = agent.NewSimpleContextManager(nil)
	}

	cm := &SmartContextManager{
		ContextManager: next,
		config:         cfg,
		store:          store,
		logger:         cfg.Logger,
		state:          agent.EmptyContextState(),
	}

	if store == nil {
		return cm, nil
	}

	loaded, messages, err := store.Load(context.Background())
	if err != nil {
		return nil, err
	}

	cm.state = loaded
	if len(messages) > 0 {
		cm.ContextManager.SetMessages(context.Background(), messages)
	}

	return cm, nil
}

func (m *SmartContextManager) Messages(ctx context.Context) []agent.Message {
	messages := m.ContextManager.Messages(ctx)
	if m.logger.Enabled(ctx, slog.LevelDebug) {
		m.logger.Debug("agent: messages updated",
			"total_messages", len(messages),
			"total_tokens", agent.EstimateMessagesTokens(messages),
		)
	}

	return messages
}

func (m *SmartContextManager) ClearMessages(ctx context.Context) {
	m.ContextManager.ClearMessages(ctx)

	m.mu.Lock()
	defer m.mu.Unlock()

	now := time.Now().UTC()
	m.state = agent.ContextManagerState{UpdatedAt: now}
	m.lastCompactedAt = time.Time{}
	m.saveLocked(ctx, nil)
}

func (m *SmartContextManager) BuildMessages(ctx context.Context) []agent.Message {
	working := cloneMessages(m.ContextManager.BuildMessages(ctx))

	if !m.config.Enabled {
		m.logger.Debug("context manager: compaction disabled",
			"messages", len(working),
		)

		return working
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	keepRounds := m.config.MaxRecentMessages
	if keepRounds <= 0 {
		keepRounds = defaultContextMaxRecentMessages
	}

	totalTokens := 0
	tokenLimitExceeded := false

	if m.config.SoftTokenLimit > 0 {
		totalTokens = agent.EstimateMessagesTokens(working)
		tokenLimitExceeded = totalTokens > m.config.SoftTokenLimit
	}

	totalRounds := countUniqueRounds(working)
	roundOverflow := totalRounds > keepRounds
	inCooldown := !m.lastCompactedAt.IsZero() && time.Since(m.lastCompactedAt) < compactionCooldown

	shouldCompact := (roundOverflow || (tokenLimitExceeded && totalRounds > keepRounds/2)) && !inCooldown
	if m.logger.Enabled(ctx, slog.LevelDebug) {
		m.logger.Debug("context manager: build messages",
			"messages", len(working),
			"rounds", totalRounds,
			"tokens", totalTokens,
			"max_recent_rounds", keepRounds,
			"soft_token_limit", m.config.SoftTokenLimit,
			"round_overflow", roundOverflow,
			"token_limit_exceeded", tokenLimitExceeded,
			"in_cooldown", inCooldown,
			"should_compact", shouldCompact,
		)
	}

	if shouldCompact {
		cut := findCutIndexForRounds(working, keepRounds)
		cut = adjustCompactionCut(working, cut)

		lastAssistantIdx := findLastAssistantMessageIndex(working)

		var overflow, retained []agent.Message

		if lastAssistantIdx > 0 && lastAssistantIdx < cut {
			overflow = cloneMessages(working[:lastAssistantIdx])
			retained = cloneMessages(working[lastAssistantIdx:])
		} else {
			overflow = cloneMessages(working[:cut])
			retained = cloneMessages(working[cut:])
		}

		if len(overflow) > 0 {
			m.logger.Debug("context manager: summarize start",
				"overflow_messages", len(overflow),
				"overflow_rounds", countUniqueRounds(overflow),
				"overflow_tokens", agent.EstimateMessagesTokens(overflow),
				"last_assistant_idx", lastAssistantIdx,
			)
			summary, err := m.config.Summarizer.Summarize(ctx, overflow)

			summary = strings.TrimSpace(summary)
			if err == nil && summary != "" {
				summaryMsg := agent.Message{
					Role:    agent.RoleUser,
					Content: &agent.Content{Text: &summary},
				}
				working = append([]agent.Message{summaryMsg}, retained...)

				now := time.Now().UTC()
				m.state.Summary = ""
				m.state.CompactionCount++
				m.state.UpdatedAt = now
				m.lastCompactedAt = now

				m.logger.Debug("context manager: summarize complete",
					"overflow_messages", len(overflow),
					"overflow_rounds", countUniqueRounds(overflow),
					"retained_messages", len(retained),
					"retained_rounds", countUniqueRounds(retained),
					"summary_chars", len(summary),
					"compaction_count", m.state.CompactionCount,
				)

				m.ContextManager.SetMessages(ctx, working)

				if m.onCompaction != nil {
					m.onCompaction()
				}
			} else {
				m.logger.Debug("context manager: summarize skipped",
					"error", err,
					"summary_empty", summary == "",
					"overflow_messages", len(overflow),
				)
			}
		} else {
			m.logger.Debug("context manager: summarize skipped",
				"reason", "empty_overflow_after_adjust",
			)
		}
	}

	m.saveLocked(ctx, working)

	return cloneMessages(working)
}

func (m *SmartContextManager) Snapshot() agent.ContextManagerState {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return agent.CopyContextState(m.state)
}

func (m *SmartContextManager) OnCompaction(fn func()) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.onCompaction = fn
}

func (m *SmartContextManager) saveLocked(ctx context.Context, messages []agent.Message) {
	if m.store == nil {
		return
	}

	var maxRI int64
	for i := range messages {
		if ri := int64(messages[i].RoundIndex); ri > maxRI {
			maxRI = ri
		}
	}

	m.state.RoundIndex = maxRI
	if err := m.store.Save(ctx, m.state, messages); err != nil {
		m.logger.Debug("context manager: save failed", "error", err)
	}
}

func adjustCompactionCut(messages []agent.Message, cut int) int {
	if cut <= 0 || cut >= len(messages) {
		return cut
	}

	overflowRoundIndexes := map[int]struct{}{}
	overflowToolUseIDs := map[string]struct{}{}

	for i := range cut {
		msg := messages[i]
		if msg.RoundIndex != 0 {
			overflowRoundIndexes[msg.RoundIndex] = struct{}{}
		}

		if msg.ToolCall != nil && msg.ToolCall.ID != "" {
			overflowToolUseIDs[msg.ToolCall.ID] = struct{}{}
		}
	}

	for cut < len(messages) {
		msg := messages[cut]

		if msg.RoundIndex != 0 {
			if _, ok := overflowRoundIndexes[msg.RoundIndex]; ok {
				if msg.ToolCall != nil && msg.ToolCall.ID != "" {
					overflowToolUseIDs[msg.ToolCall.ID] = struct{}{}
				}

				cut++

				continue
			}
		}

		if msg.Role == agent.RoleTool && msg.ToolUseID != nil {
			if _, ok := overflowToolUseIDs[*msg.ToolUseID]; ok {
				cut++
				continue
			}
		}

		break
	}

	return cut
}

func countUniqueRounds(messages []agent.Message) int {
	seen := make(map[int]struct{})

	for _, msg := range messages {
		if msg.RoundIndex != 0 {
			seen[msg.RoundIndex] = struct{}{}
		}
	}

	return len(seen)
}

func findLastAssistantMessageIndex(messages []agent.Message) int {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == agent.RoleAssistant {
			return i
		}
	}

	return -1
}

func findLastAssistantRoundIndex(messages []agent.Message) int {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == agent.RoleAssistant {
			return messages[i].RoundIndex
		}
	}

	return 0
}

func excludeMessagesByRound(messages []agent.Message, roundIndex int) []agent.Message {
	if roundIndex == 0 {
		return messages
	}

	var result []agent.Message

	for _, msg := range messages {
		if msg.RoundIndex != roundIndex {
			result = append(result, msg)
		}
	}

	return result
}

func findCutIndexForRounds(messages []agent.Message, keepRounds int) int {
	if keepRounds <= 0 || len(messages) == 0 {
		return 0
	}

	roundSet := make(map[int]struct{})

	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].RoundIndex != 0 {
			roundSet[messages[i].RoundIndex] = struct{}{}
		}

		if len(roundSet) > keepRounds {
			return i + 1
		}
	}

	return 0
}

func mergeDefaultContextManagerConfig(cfg ContextManagerConfig) ContextManagerConfig {
	if cfg.MaxRecentMessages <= 0 {
		cfg.MaxRecentMessages = defaultContextMaxRecentMessages
	}

	if cfg.SoftTokenLimit <= 0 {
		cfg.SoftTokenLimit = defaultContextSoftTokenLimit
	}

	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}

	return cfg
}
