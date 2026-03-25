package claw

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/looplj/axonhub/axon/agent"
)

const (
	defaultMaxRecentRounds    = 60
	defaultMediumTermCapacity = 10
	defaultLongTermCapacity   = 50
	defaultSoftTokenLimit     = 160_000
	defaultCompactionRatio    = 0.75
)

type HierarchicalContextManagerConfig struct {
	Enabled            bool
	MaxRecentRounds    int
	SoftTokenLimit     int
	MediumTermCapacity int
	LongTermCapacity   int
	CompactionRatio    float64
	Summarizer         *SmartSummarizer
	Logger             *slog.Logger
}

func DefaultHierarchicalContextManagerConfig() HierarchicalContextManagerConfig {
	return HierarchicalContextManagerConfig{
		Enabled:            true,
		MaxRecentRounds:    defaultMaxRecentRounds,
		SoftTokenLimit:     defaultSoftTokenLimit,
		MediumTermCapacity: defaultMediumTermCapacity,
		LongTermCapacity:   defaultLongTermCapacity,
		CompactionRatio:    defaultCompactionRatio,
	}
}

type HierarchicalContextManager struct {
	mu     sync.RWMutex
	config HierarchicalContextManagerConfig
	store  ContextManagerStore
	logger *slog.Logger

	messages      []agent.Message
	memory        MemoryState
	roundIndex    int64
	lastCompacted time.Time
	onCompaction  func()
}

func NewHierarchicalContextManager(config HierarchicalContextManagerConfig, store ContextManagerStore) (*HierarchicalContextManager, error) {
	cfg := mergeHierarchicalConfig(config)

	cm := &HierarchicalContextManager{
		config:   cfg,
		store:    store,
		logger:   cfg.Logger,
		messages: make([]agent.Message, 0),
		memory:   newMemoryState(),
	}

	if store == nil {
		return cm, nil
	}

	loaded, messages, err := store.Load(context.Background())
	if err != nil {
		return nil, err
	}

	cm.memory = loadedToMemoryState(loaded)
	if len(messages) > 0 {
		cm.messages = messages
		for _, msg := range messages {
			if int64(msg.RoundIndex) > cm.roundIndex {
				cm.roundIndex = int64(msg.RoundIndex)
			}
		}
	}

	return cm, nil
}

func loadedToMemoryState(state agent.ContextManagerState) MemoryState {
	ms := newMemoryState()
	ms.UpdatedAt = state.UpdatedAt

	if state.Summary != "" {
		ms.MediumTerm = append(ms.MediumTerm, MemoryEntry{
			ID:        "legacy-summary",
			Layer:     MemoryLayerMedium,
			Content:   state.Summary,
			CreatedAt: state.UpdatedAt,
			Metadata: MemoryMeta{
				Type: "legacy_summary",
			},
		})
	}

	return ms
}

func (m *HierarchicalContextManager) AddMessages(ctx context.Context, msgs ...agent.Message) {
	if len(msgs) == 0 {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.messages = append(m.messages, msgs...)

	for _, msg := range msgs {
		if int64(msg.RoundIndex) > m.roundIndex {
			m.roundIndex = int64(msg.RoundIndex)
		}
	}

	m.saveLocked(ctx, m.messages)
}

func (m *HierarchicalContextManager) SetMessages(ctx context.Context, msgs []agent.Message) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.messages = cloneMessages(msgs)

	m.roundIndex = 0
	for _, msg := range msgs {
		if int64(msg.RoundIndex) > m.roundIndex {
			m.roundIndex = int64(msg.RoundIndex)
		}
	}

	m.saveLocked(ctx, m.messages)
}

func (m *HierarchicalContextManager) Messages(ctx context.Context) []agent.Message {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return cloneMessages(m.messages)
}

func (m *HierarchicalContextManager) ClearMessages(ctx context.Context) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.messages = nil
	m.memory = newMemoryState()
	m.roundIndex = 0
	m.lastCompacted = time.Time{}

	m.saveLocked(ctx, nil)
}

func (m *HierarchicalContextManager) BuildMessages(ctx context.Context) []agent.Message {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.config.Enabled {
		return cloneMessages(m.messages)
	}

	working := cloneMessages(m.messages)

	totalTokens := agent.EstimateMessagesTokens(working)
	totalRounds := countUniqueRounds(working)

	shouldCompact := m.shouldCompactLocked(totalTokens, totalRounds)

	if m.logger.Enabled(ctx, slog.LevelDebug) {
		m.logger.Debug("hierarchical context manager: build messages",
			"messages", len(working),
			"rounds", totalRounds,
			"tokens", totalTokens,
			"should_compact", shouldCompact,
			"medium_term_entries", len(m.memory.MediumTerm),
			"long_term_entries", len(m.memory.LongTerm),
		)
	}

	if shouldCompact {
		working = m.performCompactionLocked(ctx, working)
	}

	m.saveLocked(ctx, working)

	return working
}

func (m *HierarchicalContextManager) shouldCompactLocked(totalTokens int, totalRounds int) bool {
	if !m.lastCompacted.IsZero() && time.Since(m.lastCompacted) < compactionCooldown {
		return false
	}

	if totalRounds > m.config.MaxRecentRounds {
		return true
	}

	if m.config.SoftTokenLimit > 0 && totalTokens > int(float64(m.config.SoftTokenLimit)*m.config.CompactionRatio) {
		return totalRounds > m.config.MaxRecentRounds/2
	}

	return false
}

func (m *HierarchicalContextManager) performCompactionLocked(ctx context.Context, working []agent.Message) []agent.Message {
	keepRounds := m.config.MaxRecentRounds
	if keepRounds <= 0 {
		keepRounds = defaultMaxRecentRounds
	}

	cut := findCutIndexForRounds(working, keepRounds)
	cut = adjustCompactionCut(working, cut)

	if cut <= 0 {
		return working
	}

	overflow := cloneMessages(working[:cut])
	retained := cloneMessages(working[cut:])

	if len(overflow) == 0 {
		return working
	}

	m.logger.Debug("hierarchical context manager: compaction start",
		"overflow_messages", len(overflow),
		"overflow_rounds", countUniqueRounds(overflow),
		"overflow_tokens", agent.EstimateMessagesTokens(overflow),
	)

	summary, err := m.createStructuredSummary(ctx, overflow)
	if err != nil {
		m.logger.Debug("hierarchical context manager: summarization failed", "error", err)
		return working
	}

	entry := m.createMemoryEntry(summary, overflow)

	m.memory.MediumTerm = append(m.memory.MediumTerm, entry)

	if len(m.memory.MediumTerm) > m.config.MediumTermCapacity {
		m.promoteToLongTermLocked()
	}

	m.memory.UpdatedAt = time.Now().UTC()
	m.lastCompacted = time.Now().UTC()

	m.logger.Debug("hierarchical context manager: compaction complete",
		"retained_messages", len(retained),
		"retained_rounds", countUniqueRounds(retained),
		"medium_term_count", len(m.memory.MediumTerm),
		"long_term_count", len(m.memory.LongTerm),
	)

	if m.onCompaction != nil {
		m.onCompaction()
	}

	return m.buildFinalMessagesLocked(retained)
}

func (m *HierarchicalContextManager) createStructuredSummary(ctx context.Context, messages []agent.Message) (*StructuredSummary, error) {
	if m.config.Summarizer == nil {
		return nil, fmt.Errorf("no summarizer configured")
	}

	return m.config.Summarizer.SummarizeStructured(ctx, messages)
}

func (m *HierarchicalContextManager) createMemoryEntry(summary *StructuredSummary, messages []agent.Message) MemoryEntry {
	if m.config.Summarizer == nil {
		return MemoryEntry{
			ID:        fmt.Sprintf("summary-%d", time.Now().Unix()),
			Layer:     MemoryLayerMedium,
			Content:   summary.Summary,
			CreatedAt: time.Now().UTC(),
		}
	}

	var (
		roundRange [2]int
		sourceIDs  []string
	)

	if len(messages) > 0 {
		minRound := messages[0].RoundIndex

		maxRound := messages[0].RoundIndex
		for _, msg := range messages {
			if msg.RoundIndex > 0 {
				if msg.RoundIndex < minRound || minRound == 0 {
					minRound = msg.RoundIndex
				}

				if msg.RoundIndex > maxRound {
					maxRound = msg.RoundIndex
				}
			}
		}

		roundRange = [2]int{minRound, maxRound}
	}

	return m.config.Summarizer.CreateMemoryEntry(summary, roundRange, sourceIDs)
}

func (m *HierarchicalContextManager) promoteToLongTermLocked() {
	if len(m.memory.MediumTerm) <= m.config.MediumTermCapacity {
		return
	}

	excess := len(m.memory.MediumTerm) - m.config.MediumTermCapacity
	toPromote := m.memory.MediumTerm[:excess]
	m.memory.MediumTerm = m.memory.MediumTerm[excess:]

	for _, entry := range toPromote {
		longEntry := MemoryEntry{
			ID:        entry.ID + "-long",
			Layer:     MemoryLayerLong,
			Content:   m.compressForLongTerm(entry),
			CreatedAt: entry.CreatedAt,
			Metadata:  entry.Metadata,
		}
		m.memory.LongTerm = append(m.memory.LongTerm, longEntry)
	}

	if len(m.memory.LongTerm) > m.config.LongTermCapacity {
		m.memory.LongTerm = m.memory.LongTerm[len(m.memory.LongTerm)-m.config.LongTermCapacity:]
	}
}

func (m *HierarchicalContextManager) compressForLongTerm(entry MemoryEntry) string {
	var sb strings.Builder

	if len(entry.Metadata.Decisions) > 0 {
		sb.WriteString("Key Decisions:\n")

		for _, d := range entry.Metadata.Decisions {
			fmt.Fprintf(&sb, "- %s: %s\n", d.Topic, d.Decision)
		}
	}

	if len(entry.Metadata.UserPrefs) > 0 {
		sb.WriteString("\nUser Preferences:\n")

		for _, up := range entry.Metadata.UserPrefs {
			fmt.Fprintf(&sb, "- %s: %s\n", up.Key, up.Value)
		}
	}

	if len(entry.Metadata.FileChanges) > 0 {
		sb.WriteString("\nFiles Modified:\n")

		fileSet := make(map[string]string)
		for _, fc := range entry.Metadata.FileChanges {
			fileSet[fc.Path] = fc.Operation
		}

		for path, op := range fileSet {
			fmt.Fprintf(&sb, "- %s (%s)\n", path, op)
		}
	}

	if sb.Len() == 0 {
		if len(entry.Content) > 500 {
			return entry.Content[:500] + "..."
		}

		return entry.Content
	}

	return sb.String()
}

func (m *HierarchicalContextManager) buildFinalMessagesLocked(retained []agent.Message) []agent.Message {
	var result []agent.Message

	if len(m.memory.LongTerm) > 0 {
		longTermContent := m.formatLongTermMemory()
		if longTermContent != "" {
			result = append(result, agent.Message{
				Role:    agent.RoleUser,
				Content: &agent.Content{Text: &longTermContent},
			})
		}
	}

	if len(m.memory.MediumTerm) > 0 {
		for _, entry := range m.memory.MediumTerm {
			result = append(result, entry.ToMessage())
		}
	}

	result = append(result, retained...)

	return result
}

func (m *HierarchicalContextManager) formatLongTermMemory() string {
	if len(m.memory.LongTerm) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("## Persistent Memory\n\n")
	sb.WriteString("The following information has been retained from earlier conversations:\n\n")

	for _, entry := range m.memory.LongTerm {
		if entry.Content != "" {
			sb.WriteString(entry.Content)
			sb.WriteString("\n\n")
		}
	}

	return sb.String()
}

func (m *HierarchicalContextManager) Snapshot() agent.ContextManagerState {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.snapshotLocked()
}

func (m *HierarchicalContextManager) snapshotLocked() agent.ContextManagerState {
	var summary string

	if len(m.memory.MediumTerm) > 0 {
		var sb strings.Builder
		for _, entry := range m.memory.MediumTerm {
			sb.WriteString(entry.Content)
			sb.WriteString("\n\n")
		}

		summary = sb.String()
	}

	return agent.ContextManagerState{
		Summary:         summary,
		CompactionCount: int64(len(m.memory.MediumTerm) + len(m.memory.LongTerm)),
		RoundIndex:      m.roundIndex,
		UpdatedAt:       m.memory.UpdatedAt,
	}
}

func (m *HierarchicalContextManager) OnCompaction(fn func()) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.onCompaction = fn
}

func (m *HierarchicalContextManager) MemoryState() MemoryState {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.memory.Clone()
}

func (m *HierarchicalContextManager) AddLongTermMemory(entry MemoryEntry) {
	m.mu.Lock()
	defer m.mu.Unlock()

	entry.Layer = MemoryLayerLong
	entry.CreatedAt = time.Now().UTC()
	m.memory.LongTerm = append(m.memory.LongTerm, entry)

	if len(m.memory.LongTerm) > m.config.LongTermCapacity {
		m.memory.LongTerm = m.memory.LongTerm[len(m.memory.LongTerm)-m.config.LongTermCapacity:]
	}

	m.memory.UpdatedAt = time.Now().UTC()
}

func (m *HierarchicalContextManager) saveLocked(ctx context.Context, messages []agent.Message) {
	if m.store == nil {
		return
	}

	var maxRI int64
	for i := range messages {
		if ri := int64(messages[i].RoundIndex); ri > maxRI {
			maxRI = ri
		}
	}

	m.roundIndex = maxRI

	state := m.snapshotLocked()
	state.RoundIndex = maxRI

	if err := m.store.Save(ctx, state, messages); err != nil {
		m.logger.Debug("hierarchical context manager: save failed", "error", err)
	}
}

func mergeHierarchicalConfig(cfg HierarchicalContextManagerConfig) HierarchicalContextManagerConfig {
	if cfg.MaxRecentRounds <= 0 {
		cfg.MaxRecentRounds = defaultMaxRecentRounds
	}

	if cfg.SoftTokenLimit <= 0 {
		cfg.SoftTokenLimit = defaultSoftTokenLimit
	}

	if cfg.MediumTermCapacity <= 0 {
		cfg.MediumTermCapacity = defaultMediumTermCapacity
	}

	if cfg.LongTermCapacity <= 0 {
		cfg.LongTermCapacity = defaultLongTermCapacity
	}

	if cfg.CompactionRatio <= 0 || cfg.CompactionRatio >= 1 {
		cfg.CompactionRatio = defaultCompactionRatio
	}

	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}

	return cfg
}

type HierarchicalMemoryStore struct {
	dir string
	mu  sync.Mutex
}

func NewHierarchicalMemoryStore(dir string) *HierarchicalMemoryStore {
	return &HierarchicalMemoryStore{dir: dir}
}

func (s *HierarchicalMemoryStore) Load(_ context.Context) (agent.ContextManagerState, []agent.Message, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	stateData, err := os.ReadFile(s.statePath())
	if err != nil {
		if os.IsNotExist(err) {
			return agent.ContextManagerState{}, nil, nil
		}

		return agent.ContextManagerState{}, nil, fmt.Errorf("read state file: %w", err)
	}

	var state agent.ContextManagerState
	if err := json.Unmarshal(stateData, &state); err != nil {
		return agent.ContextManagerState{}, nil, fmt.Errorf("unmarshal state: %w", err)
	}

	messagesData, err := os.ReadFile(s.messagesPath())
	if err != nil {
		if os.IsNotExist(err) {
			return state, nil, nil
		}

		return agent.ContextManagerState{}, nil, fmt.Errorf("read messages file: %w", err)
	}

	var messages []agent.Message
	if err := json.Unmarshal(messagesData, &messages); err != nil {
		return agent.ContextManagerState{}, nil, fmt.Errorf("unmarshal messages: %w", err)
	}

	return state, messages, nil
}

func (s *HierarchicalMemoryStore) Save(_ context.Context, state agent.ContextManagerState, messages []agent.Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return fmt.Errorf("create directory: %w", err)
	}

	stateData, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("marshal state: %w", err)
	}

	if err := writeFileAtomic(s.statePath(), stateData); err != nil {
		return fmt.Errorf("write state: %w", err)
	}

	messagesData, err := json.Marshal(messages)
	if err != nil {
		return fmt.Errorf("marshal messages: %w", err)
	}

	if err := writeFileAtomic(s.messagesPath(), messagesData); err != nil {
		return fmt.Errorf("write messages: %w", err)
	}

	return nil
}

func (s *HierarchicalMemoryStore) statePath() string {
	return filepath.Join(s.dir, "hierarchical_state.json")
}

func (s *HierarchicalMemoryStore) messagesPath() string {
	return filepath.Join(s.dir, "messages.json")
}

func writeFileAtomic(path string, data []byte) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}

	return os.Rename(tmp, path)
}

var (
	_ agent.ContextManager = (*HierarchicalContextManager)(nil)
	_ ContextManagerStore  = (*HierarchicalMemoryStore)(nil)
)
