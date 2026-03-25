package claw

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/looplj/axonhub/axon/agent"
	"github.com/looplj/axonhub/axon/bus"
	"github.com/looplj/axonhub/axon/subagent"
)

const (
	summarizerPromptTemplate = `You are a context summarization assistant. Your task is to analyze the conversation history and create a structured summary that preserves critical information.

IMPORTANT: You must output ONLY valid JSON, no other text.

Analyze the following conversation and extract:
1. A concise narrative summary of what happened
2. Key decisions made and why
3. File changes (created, modified, deleted)
4. User preferences discovered
5. Important entities (people, projects, concepts)
6. Task progress status

Format your response as JSON with this exact structure:
{
  "summary": "A concise narrative summary of the conversation",
  "decisions": [
    {"topic": "what was decided", "decision": "the decision", "reason": "why", "round": 0}
  ],
  "file_changes": [
    {"path": "file path", "operation": "created|modified|deleted", "summary": "what changed", "round": 0}
  ],
  "user_prefs": [
    {"key": "preference name", "value": "preference value", "round": 0}
  ],
  "entities": [
    {"name": "entity name", "type": "person|project|concept|tool", "desc": "description"}
  ],
  "task_progress": [
    {"task_id": "task identifier", "status": "pending|in_progress|completed|blocked", "progress": "current state", "updated_at": 0}
  ]
}

Conversation to summarize:

%s`
)

type StructuredSummary struct {
	Summary      string             `json:"summary"`
	Decisions    []DecisionInfo     `json:"decisions"`
	FileChanges  []FileChangeInfo   `json:"file_changes"`
	UserPrefs    []UserPrefInfo     `json:"user_prefs"`
	Entities     []EntityInfo       `json:"entities"`
	TaskProgress []TaskProgressInfo `json:"task_progress"`
}

type SmartSummarizerConfig struct {
	MaxSummaryLength    int
	ExtractEntities     bool
	ExtractDecisions    bool
	ExtractFileChanges  bool
	ExtractUserPrefs    bool
	ExtractTaskProgress bool
}

func DefaultSmartSummarizerConfig() SmartSummarizerConfig {
	return SmartSummarizerConfig{
		MaxSummaryLength:    4000,
		ExtractEntities:     true,
		ExtractDecisions:    true,
		ExtractFileChanges:  true,
		ExtractUserPrefs:    true,
		ExtractTaskProgress: true,
	}
}

type SmartSummarizer struct {
	provider    agent.Provider
	config      SmartSummarizerConfig
	toolSource  subagent.ToolSource
	bus         bus.EventBus
	middlewares []agent.Middleware
	logger      *slog.Logger
}

func NewSmartSummarizer(provider agent.Provider, config SmartSummarizerConfig) *SmartSummarizer {
	return &SmartSummarizer{
		provider: provider,
		config:   config,
		logger:   slog.Default(),
	}
}

func NewSmartSummarizerWithTools(provider agent.Provider, config SmartSummarizerConfig, toolSource subagent.ToolSource, bus bus.EventBus, middlewares []agent.Middleware, logger *slog.Logger) *SmartSummarizer {
	if logger == nil {
		logger = slog.Default()
	}

	return &SmartSummarizer{
		provider:    provider,
		config:      config,
		toolSource:  toolSource,
		bus:         bus,
		middlewares: middlewares,
		logger:      logger,
	}
}

func (s *SmartSummarizer) Summarize(ctx context.Context, messages []agent.Message) (string, error) {
	if len(messages) == 0 {
		return "", nil
	}

	conversationText := s.buildConversationText(messages)

	if s.toolSource != nil {
		return s.summarizeWithSubagent(ctx, conversationText, messages)
	}

	return s.summarizeDirect(ctx, conversationText, messages)
}

func (s *SmartSummarizer) summarizeWithSubagent(ctx context.Context, conversationText string, messages []agent.Message) (string, error) {
	systemPrompt := `You are a context summarization assistant. Your task is to analyze the conversation history and create a structured summary that preserves critical information.

IMPORTANT: You must output ONLY valid JSON, no other text.

Analyze the following conversation and extract:
1. A concise narrative summary of what happened
2. Key decisions made and why
3. File changes (created, modified, deleted)
4. User preferences discovered
5. Important entities (people, projects, concepts)
6. Task progress status

Format your response as JSON with this exact structure:
{
  "summary": "A concise narrative summary of the conversation",
  "decisions": [
    {"topic": "what was decided", "decision": "the decision", "reason": "why", "round": 0}
  ],
  "file_changes": [
    {"path": "file path", "operation": "created|modified|deleted", "summary": "what changed", "round": 0}
  ],
  "user_prefs": [
    {"key": "preference name", "value": "preference value", "round": 0}
  ],
  "entities": [
    {"name": "entity name", "type": "person|project|concept|tool", "desc": "description"}
  ],
  "task_progress": [
    {"task_id": "task identifier", "status": "pending|in_progress|completed|blocked", "progress": "current state", "updated_at": 0}
  ]
}

You have access to the Skill tool. Use the "memory" skill if you need to store or retrieve memories that are relevant to this summarization task.`

	task := fmt.Sprintf("Please summarize the following conversation:\n\n%s", conversationText)

	result, err := subagent.Run(ctx, subagent.Config{
		Model:         "",
		SystemPrompts: []string{systemPrompt},
		AllowedTools:  []string{"Skill"},
		Provider:      s.provider,
		Bus:           s.bus,
		Middlewares:   s.middlewares,
		Logger:        s.logger.With("component", "summarizer_subagent"),
	}, task, s.toolSource)
	if err != nil {
		s.logger.Warn("summarizer subagent failed, falling back to direct summarization", "error", err)
		return s.summarizeDirect(ctx, conversationText, messages)
	}

	if result.Output == "" {
		return "", fmt.Errorf("summarizer subagent returned empty output")
	}

	summary, err := s.parseStructuredSummary(result.Output)
	if err != nil {
		return "", fmt.Errorf("failed to parse structured summary: %w", err)
	}

	return s.formatSummaryForContext(summary), nil
}

func (s *SmartSummarizer) summarizeDirect(ctx context.Context, conversationText string, messages []agent.Message) (string, error) {
	prompt := fmt.Sprintf(summarizerPromptTemplate, conversationText)

	resp, err := s.provider.Chat(ctx, "", nil, []agent.Message{
		{Role: agent.RoleUser, Content: &agent.Content{Text: &prompt}},
	})
	if err != nil {
		return "", fmt.Errorf("summarization failed: %w", err)
	}

	var text string

	for _, msg := range resp.Messages {
		if msg.Content != nil && msg.Content.Text != nil {
			text = *msg.Content.Text
			break
		}
	}

	if text == "" {
		return "", fmt.Errorf("summarization returned empty response")
	}

	summary, err := s.parseStructuredSummary(text)
	if err != nil {
		return "", fmt.Errorf("failed to parse structured summary: %w", err)
	}

	return s.formatSummaryForContext(summary), nil
}

func (s *SmartSummarizer) SummarizeStructured(ctx context.Context, messages []agent.Message) (*StructuredSummary, error) {
	if len(messages) == 0 {
		return &StructuredSummary{}, nil
	}

	conversationText := s.buildConversationText(messages)

	if s.toolSource != nil {
		return s.summarizeStructuredWithSubagent(ctx, conversationText, messages)
	}

	return s.summarizeStructuredDirect(ctx, conversationText, messages)
}

func (s *SmartSummarizer) summarizeStructuredWithSubagent(ctx context.Context, conversationText string, messages []agent.Message) (*StructuredSummary, error) {
	systemPrompt := `You are a context summarization assistant. Your task is to analyze the conversation history and create a structured summary that preserves critical information.

IMPORTANT: You must output ONLY valid JSON, no other text.

Analyze the following conversation and extract:
1. A concise narrative summary of what happened
2. Key decisions made and why
3. File changes (created, modified, deleted)
4. User preferences discovered
5. Important entities (people, projects, concepts)
6. Task progress status

Format your response as JSON with this exact structure:
{
  "summary": "A concise narrative summary of the conversation",
  "decisions": [
    {"topic": "what was decided", "decision": "the decision", "reason": "why", "round": 0}
  ],
  "file_changes": [
    {"path": "file path", "operation": "created|modified|deleted", "summary": "what changed", "round": 0}
  ],
  "user_prefs": [
    {"key": "preference name", "value": "preference value", "round": 0}
  ],
  "entities": [
    {"name": "entity name", "type": "person|project|concept|tool", "desc": "description"}
  ],
  "task_progress": [
    {"task_id": "task identifier", "status": "pending|in_progress|completed|blocked", "progress": "current state", "updated_at": 0}
  ]
}

You have access to the Skill tool. Use the "memory" skill if you need to store or retrieve memories that are relevant to this summarization task.`

	task := fmt.Sprintf("Please summarize the following conversation:\n\n%s", conversationText)

	result, err := subagent.Run(ctx, subagent.Config{
		Model:         "",
		SystemPrompts: []string{systemPrompt},
		AllowedTools:  []string{"Skill"},
		Provider:      s.provider,
		Bus:           s.bus,
		Middlewares:   s.middlewares,
		Logger:        s.logger.With("component", "summarizer_subagent"),
	}, task, s.toolSource)
	if err != nil {
		s.logger.Warn("summarizer subagent failed, falling back to direct summarization", "error", err)
		return s.summarizeStructuredDirect(ctx, conversationText, messages)
	}

	if result.Output == "" {
		return nil, fmt.Errorf("summarizer subagent returned empty output")
	}

	summary, err := s.parseStructuredSummary(result.Output)
	if err != nil {
		return nil, fmt.Errorf("failed to parse structured summary: %w", err)
	}

	return summary, nil
}

func (s *SmartSummarizer) summarizeStructuredDirect(ctx context.Context, conversationText string, messages []agent.Message) (*StructuredSummary, error) {
	prompt := fmt.Sprintf(summarizerPromptTemplate, conversationText)

	resp, err := s.provider.Chat(ctx, "", nil, []agent.Message{
		{Role: agent.RoleUser, Content: &agent.Content{Text: &prompt}},
	})
	if err != nil {
		return nil, fmt.Errorf("summarization failed: %w", err)
	}

	var text string

	for _, msg := range resp.Messages {
		if msg.Content != nil && msg.Content.Text != nil {
			text = *msg.Content.Text
			break
		}
	}

	if text == "" {
		return nil, fmt.Errorf("summarization returned empty response")
	}

	summary, err := s.parseStructuredSummary(text)
	if err != nil {
		return nil, fmt.Errorf("failed to parse structured summary: %w", err)
	}

	return summary, nil
}

func (s *SmartSummarizer) buildConversationText(messages []agent.Message) string {
	var sb strings.Builder

	for _, msg := range messages {
		switch msg.Role {
		case agent.RoleUser:
			sb.WriteString("USER: ")
		case agent.RoleAssistant:
			sb.WriteString("ASSISTANT: ")
		case agent.RoleTool:
			sb.WriteString("TOOL RESULT: ")
		case agent.RoleSystem:
			sb.WriteString("SYSTEM: ")
		}

		if msg.Content != nil {
			sb.WriteString(msg.Content.String())
		}

		if msg.ToolCall != nil {
			fmt.Fprintf(&sb, " [Tool: %s, Args: %s]", msg.ToolCall.Name, msg.ToolCall.Input)
		}

		sb.WriteString("\n\n")
	}

	return sb.String()
}

func (s *SmartSummarizer) parseStructuredSummary(text string) (*StructuredSummary, error) {
	jsonStart := strings.Index(text, "{")

	jsonEnd := strings.LastIndex(text, "}")
	if jsonStart == -1 || jsonEnd == -1 || jsonEnd <= jsonStart {
		return nil, fmt.Errorf("no valid JSON found in response")
	}

	jsonStr := text[jsonStart : jsonEnd+1]

	var summary StructuredSummary
	if err := json.Unmarshal([]byte(jsonStr), &summary); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	return &summary, nil
}

func (s *SmartSummarizer) formatSummaryForContext(summary *StructuredSummary) string {
	var sb strings.Builder

	sb.WriteString("## Conversation Summary\n\n")
	sb.WriteString(summary.Summary)
	sb.WriteString("\n\n")

	if len(summary.Decisions) > 0 && s.config.ExtractDecisions {
		sb.WriteString("### Key Decisions\n")

		for _, d := range summary.Decisions {
			fmt.Fprintf(&sb, "- **%s**: %s", d.Topic, d.Decision)

			if d.Reason != "" {
				fmt.Fprintf(&sb, " (Reason: %s)", d.Reason)
			}

			sb.WriteString("\n")
		}

		sb.WriteString("\n")
	}

	if len(summary.FileChanges) > 0 && s.config.ExtractFileChanges {
		sb.WriteString("### File Changes\n")

		for _, fc := range summary.FileChanges {
			fmt.Fprintf(&sb, "- %s: %s", fc.Operation, fc.Path)

			if fc.Summary != "" {
				fmt.Fprintf(&sb, " - %s", fc.Summary)
			}

			sb.WriteString("\n")
		}

		sb.WriteString("\n")
	}

	if len(summary.UserPrefs) > 0 && s.config.ExtractUserPrefs {
		sb.WriteString("### User Preferences\n")

		for _, up := range summary.UserPrefs {
			fmt.Fprintf(&sb, "- %s: %s\n", up.Key, up.Value)
		}

		sb.WriteString("\n")
	}

	if len(summary.Entities) > 0 && s.config.ExtractEntities {
		sb.WriteString("### Key Entities\n")

		for _, e := range summary.Entities {
			fmt.Fprintf(&sb, "- %s (%s)", e.Name, e.Type)

			if e.Desc != "" {
				fmt.Fprintf(&sb, ": %s", e.Desc)
			}

			sb.WriteString("\n")
		}

		sb.WriteString("\n")
	}

	if len(summary.TaskProgress) > 0 && s.config.ExtractTaskProgress {
		sb.WriteString("### Task Progress\n")

		for _, tp := range summary.TaskProgress {
			fmt.Fprintf(&sb, "- %s: %s", tp.TaskID, tp.Status)

			if tp.Progress != "" {
				fmt.Fprintf(&sb, " - %s", tp.Progress)
			}

			sb.WriteString("\n")
		}

		sb.WriteString("\n")
	}

	return sb.String()
}

func (s *SmartSummarizer) CreateMemoryEntry(summary *StructuredSummary, roundRange [2]int, sourceIDs []string) MemoryEntry {
	now := time.Now().UTC()

	metadata := MemoryMeta{
		Type:       "structured_summary",
		Importance: s.calculateImportance(summary),
		RoundRange: roundRange,
		TokenCount: agent.EstimateTokens(s.formatSummaryForContext(summary)),
		SourceIDs:  sourceIDs,
	}

	if s.config.ExtractDecisions {
		metadata.Decisions = summary.Decisions
	}

	if s.config.ExtractFileChanges {
		metadata.FileChanges = summary.FileChanges
	}

	if s.config.ExtractUserPrefs {
		metadata.UserPrefs = summary.UserPrefs
	}

	if s.config.ExtractEntities {
		metadata.Entities = summary.Entities
	}

	if s.config.ExtractTaskProgress {
		metadata.TaskProgress = summary.TaskProgress
	}

	return MemoryEntry{
		ID:        fmt.Sprintf("summary-%d-%d", roundRange[0], roundRange[1]),
		Layer:     MemoryLayerMedium,
		Content:   s.formatSummaryForContext(summary),
		CreatedAt: now,
		Metadata:  metadata,
	}
}

func (s *SmartSummarizer) calculateImportance(summary *StructuredSummary) int {
	importance := 5

	if len(summary.Decisions) > 0 {
		importance += 2
	}

	if len(summary.FileChanges) > 0 {
		importance += 1
	}

	if len(summary.UserPrefs) > 0 {
		importance += 2
	}

	if len(summary.TaskProgress) > 0 {
		importance += 1
	}

	if importance > 10 {
		importance = 10
	}

	return importance
}
