import React, { useState, useRef, useEffect } from 'react';
import {
  Compass,
  ChevronDown,
  ChevronRight,
  School,
  Building,
  CheckCircle2,
  Send,
  Zap,
  RotateCcw,
  Terminal,
  Sparkles,
} from 'lucide-react';

export interface LogMessage {
  id: string;
  type: 'agent' | 'spatial' | 'system';
  title: string;
  detail: string;
  timestamp: string;
  data?: any;
}

export interface StructuredAgentData {
  target_name?: string;
  school_name?: string;
  radius_meters?: number;
  total_blocks_found?: number;
  qualifying_blocks?: string[];
  nearby_amenities?: any[];
  market_summary?: any;
  financial_summary?: any;
  policy_notes?: string[];
  school_lookup?: { status: string; schools: { name: string; distance_meters: number }[] };
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'agent';
  text: string;
  timestamp: string;
  status?: 'running' | 'completed' | 'error';
  structuredData?: StructuredAgentData;
  stepLogs?: LogMessage[];
}

export interface ModelInfo {
  profile?: string;
  model?: string;
  mode?: string;
  status?: string;
  message?: string;
}

interface AgentChatProps {
  messages: ChatMessage[];
  agentStatus: string;
  wsConnected: boolean;
  modelInfo?: ModelInfo;
  onSendQuery: (query: string) => void;
  onClear: () => void;
  onSelectBlock?: (blockSummary: string) => void;
}

export const AgentChat: React.FC<AgentChatProps> = ({
  messages,
  agentStatus,
  wsConnected,
  modelInfo,
  onSendQuery,
  onClear,
  onSelectBlock,
}) => {
  const [inputValue, setInputValue] = useState('');
  const [expandedTraceIds, setExpandedTraceIds] = useState<Record<string, boolean>>({});
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll on new message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, agentStatus]);

  const toggleTrace = (id: string) => {
    setExpandedTraceIds((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || !wsConnected || agentStatus === 'SEARCHING') return;
    onSendQuery(inputValue.trim());
    setInputValue('');
  };

  const handleQuickPrompt = (prompt: string) => {
    onSendQuery(prompt);
  };

  return (
    <aside className="left-panel">
      {/* Sleek Minimal Header */}
      <header className="header-bar">
        <div className="brand-badge">
          <div className="brand-icon">
            <Compass size={18} />
          </div>
          <div>
            <h1 className="brand-title">HDB-Compass</h1>
            <p className="brand-subtitle">AI Decision Support for Singapore Housing</p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div className={`status-pill ${wsConnected ? 'connected' : 'disconnected'}`}>
            <span className="status-dot" />
            <span className="status-text">{wsConnected ? 'Live' : 'Connecting'}</span>
          </div>

          <button
            onClick={onClear}
            className="clear-btn"
            title="Reset Conversation"
          >
            <RotateCcw size={12} />
            <span>Clear</span>
          </button>
        </div>
      </header>

      {/* Conversational Chat Thread */}
      <div role="status" style={{ padding: '8px 20px', fontSize: 12, color: '#94a3b8', borderBottom: '1px solid rgba(148,163,184,0.15)' }}>
        {!modelInfo ? 'Reading model configuration…' : modelInfo.status === 'fallback'
          ? `Model fallback: ${modelInfo.message}`
          : modelInfo.mode === 'deterministic'
            ? 'Rule-based mode — no language model. Use explicit search criteria.'
            : modelInfo.status === 'error' || modelInfo.status === 'unavailable'
              ? `Model unavailable: ${modelInfo.message}`
              : `Model configured: ${modelInfo.profile} / ${modelInfo.model}`}
      </div>
      <div className="chat-container">
        {/* If only 1 message (welcome), show elegant prompt suggestion cards inside welcome */}
        {messages.length <= 1 && (
          <div className="welcome-hero-card">
            <div className="welcome-header">
              <Sparkles size={16} color="#38bdf8" />
              <span>Explore Singapore Public Housing</span>
            </div>
            <p className="welcome-desc">
              Find nearby primary schools, explore approximate straight-line distances, or search historical resale data.
            </p>
            <div className="welcome-prompts-grid">
              <button
                className="welcome-prompt-card"
                onClick={() => handleQuickPrompt('Show HDB blocks within 1km of Ai Tong School.')}
              >
                <div className="prompt-card-icon school">
                  <School size={14} />
                </div>
                <div className="prompt-card-text">
                  <span className="prompt-title">Ai Tong School</span>
                  <span className="prompt-subtitle">Approximate 1km school radius</span>
                </div>
              </button>

              <button
                className="welcome-prompt-card"
                onClick={() => handleQuickPrompt('Show HDB blocks within 1km of Catholic High School.')}
              >
                <div className="prompt-card-icon school">
                  <School size={14} />
                </div>
                <div className="prompt-card-text">
                  <span className="prompt-title">Catholic High School</span>
                  <span className="prompt-subtitle">Approximate 1km school radius</span>
                </div>
              </button>

              <button
                className="welcome-prompt-card"
                onClick={() => handleQuickPrompt('Ang Mo Kio MRT 300m')}
              >
                <div className="prompt-card-icon transit">
                  <Zap size={14} />
                </div>
                <div className="prompt-card-text">
                  <span className="prompt-title">Ang Mo Kio MRT</span>
                  <span className="prompt-subtitle">300m Walk Proximity</span>
                </div>
              </button>

              <button
                className="welcome-prompt-card"
                onClick={() => handleQuickPrompt('640 Rowell Road 500m')}
              >
                <div className="prompt-card-icon location">
                  <Building size={14} />
                </div>
                <div className="prompt-card-text">
                  <span className="prompt-title">Rowell Road</span>
                  <span className="prompt-subtitle">500m Jalan Besar Buffer</span>
                </div>
              </button>
            </div>
          </div>
        )}

        {/* Messages */}
        {messages.map((msg) => (
          <div key={msg.id} className={`chat-message-row ${msg.sender}`}>
            {msg.sender === 'user' ? (
              <div className="user-message-bubble">
                <p className="message-text">{msg.text}</p>
                <span className="message-time">{msg.timestamp}</span>
              </div>
            ) : (
              <div className="agent-message-card">
                {/* Agent Header */}
                <div className="agent-card-header">
                  <div className="agent-avatar">
                    <Compass size={14} color="#38bdf8" />
                  </div>
                  <div className="agent-title-group">
                    <span className="agent-name">HDB-Compass Assistant</span>
                    <span className="agent-role-tag">Multi-Agent AI</span>
                  </div>
                  <span className="message-time" style={{ marginLeft: 'auto' }}>
                    {msg.timestamp}
                  </span>
                </div>

                {/* Loading indicator */}
                {msg.status === 'running' && (
                  <div className="agent-running-banner">
                    <div className="spinner-ring" />
                    <span>Querying SVY21 spatial index and government APIs...</span>
                  </div>
                )}

                {/* Formatted Content */}
                <div className="agent-card-body">
                  <div className="agent-summary-text">
                    {msg.text.split('\n').map((line, idx) => {
                      if (line.startsWith('**') && line.endsWith('**')) {
                        return (
                          <h4 key={idx} className="summary-heading">
                            {line.replace(/\*\*/g, '')}
                          </h4>
                        );
                      }
                      if (line.startsWith('- **') || line.startsWith('* ')) {
                        return (
                          <p key={idx} className="summary-bullet">
                            <span className="bullet-dot">•</span>
                            <span>{line.replace(/^[-*]\s*/, '').replace(/\*\*/g, '')}</span>
                          </p>
                        );
                      }
                      return line.trim() ? (
                        <p key={idx} className="summary-paragraph">
                          {line}
                        </p>
                      ) : null;
                    })}
                  </div>

                  {/* Candidate HDB Blocks Cards */}
                  {msg.structuredData?.qualifying_blocks && msg.structuredData.qualifying_blocks.length > 0 && (
                    <div className="candidate-blocks-section">
                      <div className="section-title-bar">
                        <span className="section-title">
                          <CheckCircle2 size={13} color="#10b981" />
                          Matching HDB Blocks ({msg.structuredData.total_blocks_found || msg.structuredData.qualifying_blocks.length}):
                        </span>
                        <span className="section-subtext">Click any pin on map</span>
                      </div>

                      <div className="candidate-cards-grid">
                        {msg.structuredData.qualifying_blocks.slice(0, 6).map((blockStr, bIdx) => {
                          const distMatch = blockStr.match(/\((\d+(?:\.\d+)?)m\)/);
                          const dist = distMatch ? `${distMatch[1]}m` : null;
                          const cleanName = blockStr.replace(/\s*\(\d+(?:\.\d+)?m\)/, '');

                          return (
                            <div
                              key={bIdx}
                              className="candidate-block-pill"
                              onClick={() => onSelectBlock && onSelectBlock(cleanName)}
                            >
                              <div className="block-pill-header">
                                <span className="block-code">{cleanName}</span>
                                {dist && <span className="block-distance-badge">{dist}</span>}
                              </div>
                              <div className="block-meta">
                                <span>{dist ? 'Approximate straight-line distance' : 'Distance not calculated'}</span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Nearest Transit & Amenities Pill */}
                  {msg.structuredData?.nearby_amenities && msg.structuredData.nearby_amenities.length > 0 && (
                    <div className="amenities-summary-badge">
                      <Zap size={12} color="#10b981" />
                      <span>
                        Nearest Transit: <strong>{msg.structuredData.nearby_amenities[0].name}</strong> at{' '}
                        <strong>{msg.structuredData.nearby_amenities[0].distance_meters?.toFixed(0)}m</strong>
                      </span>
                    </div>
                  )}

                  {/* Collapsible Execution Trace */}
                  {msg.stepLogs && msg.stepLogs.length > 0 && (
                    <div className="trace-accordion">
                      <button
                        className="trace-toggle-btn"
                        onClick={() => toggleTrace(msg.id)}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <Terminal size={12} color="#64748b" />
                          <span>View Execution Steps ({msg.stepLogs.length})</span>
                        </div>
                        {expandedTraceIds[msg.id] ? (
                          <ChevronDown size={14} />
                        ) : (
                          <ChevronRight size={14} />
                        )}
                      </button>

                      {expandedTraceIds[msg.id] && (
                        <div className="trace-content-box">
                          {msg.stepLogs.map((log) => (
                            <div key={log.id} className="trace-item">
                              <div className="trace-item-header">
                                <span className="trace-title">{log.title}</span>
                                <span className="trace-timestamp">{log.timestamp}</span>
                              </div>
                              <p className="trace-detail">{log.detail}</p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      {/* Clean Bottom Prompt Input Bar */}
      <footer className="chat-input-bar">
        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="text"
            className="chat-input-field"
            placeholder="Ask about schools, MRT distance, or click anywhere on the map..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            disabled={!wsConnected || agentStatus === 'SEARCHING'}
          />
          <button
            type="submit"
            className="send-btn"
            disabled={!wsConnected || !inputValue.trim() || agentStatus === 'SEARCHING'}
          >
            <Send size={14} />
            <span>Send</span>
          </button>
        </form>
        <div className="input-hint">
          <span>💡 Tip: Click anywhere on the map to pin exact coordinates</span>
        </div>
      </footer>
    </aside>
  );
};
