import React, { useState, useRef, useEffect } from 'react';
import {
  Compass,
  ChevronDown,
  ChevronRight,
  School,
  Building,
  CheckCircle2,
  Clock,
  Navigation,
  Send,
  Zap,
  RotateCcw,
  Terminal,
  Sparkles,
} from 'lucide-react';
import { Mascot3D, MascotAvatar, useMascotFavicon } from './Mascot';

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

interface AgentChatProps {
  messages: ChatMessage[];
  agentStatus: string;
  wsConnected: boolean;
  onSendQuery: (query: string) => void;
  onClear: () => void;
  onSelectBlock?: (blockSummary: string) => void;
}

export const AgentChat: React.FC<AgentChatProps> = ({
  messages,
  agentStatus,
  wsConnected,
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
    if (!inputValue.trim() || !wsConnected) return;
    onSendQuery(inputValue.trim());
    setInputValue('');
  };

  const handleQuickPrompt = (prompt: string) => {
    onSendQuery(prompt);
  };

  useMascotFavicon();

  return (
    <aside className="left-panel">
      {/* Sleek Minimal Header */}
      <header className="header-bar">
        <div className="brand-badge">
          <div className="brand-icon mascot">
            <MascotAvatar size={32} fallback={<Compass size={18} />} />
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
      <div className="chat-container">
        {/* If only 1 message (welcome), show elegant prompt suggestion cards inside welcome */}
        {messages.length <= 1 && (
          <div className="welcome-hero-card">
            <div className="welcome-mascot-row">
              <Mascot3D size={112} mood="wave" />
              <div>
                <div className="welcome-header">
                  <Sparkles size={16} color="#38bdf8" />
                  <span>Explore Singapore Public Housing</span>
                </div>
                <p className="welcome-desc">
                  Hi, I'm your HDB-Compass home guide. Ask about MOE Primary 1 registration priority distances, walking
                  commute to MRT stations, or amenity filters.
                </p>
              </div>
            </div>
            <div className="welcome-prompts-grid">
              <button
                className="welcome-prompt-card"
                onClick={() => handleQuickPrompt('Ai Tong School 1km')}
              >
                <div className="prompt-card-icon school">
                  <School size={14} />
                </div>
                <div className="prompt-card-text">
                  <span className="prompt-title">Ai Tong School</span>
                  <span className="prompt-subtitle">1km MOE Priority Boundary</span>
                </div>
              </button>

              <button
                className="welcome-prompt-card"
                onClick={() => handleQuickPrompt('Catholic High School 1km')}
              >
                <div className="prompt-card-icon school">
                  <School size={14} />
                </div>
                <div className="prompt-card-text">
                  <span className="prompt-title">Catholic High School</span>
                  <span className="prompt-subtitle">Bishan Primary 1 Buffer</span>
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
                  <div className="agent-avatar mascot">
                    <MascotAvatar size={24} fallback={<Compass size={14} color="#38bdf8" />} />
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
                    <Mascot3D size={40} mood="thinking" framing="portrait" />
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
                          Matching HDB Resale Blocks ({msg.structuredData.total_blocks_found || msg.structuredData.qualifying_blocks.length}):
                        </span>
                        <span className="section-subtext">Click any pin on map</span>
                      </div>

                      <div className="candidate-cards-grid">
                        {msg.structuredData.qualifying_blocks.slice(0, 6).map((blockStr, bIdx) => {
                          const distMatch = blockStr.match(/\((\d+)m\)/);
                          const dist = distMatch ? `${distMatch[1]}m` : '< 1000m';
                          const cleanName = blockStr.replace(/\s*\(\d+m\)/, '');
                          const walkMins = distMatch ? Math.max(1, Math.round(parseInt(distMatch[1]) / 80)) : 10;

                          return (
                            <div
                              key={bIdx}
                              className="candidate-block-pill"
                              onClick={() => onSelectBlock && onSelectBlock(cleanName)}
                            >
                              <div className="block-pill-header">
                                <span className="block-code">{cleanName}</span>
                                <span className="block-distance-badge">{dist}</span>
                              </div>
                              <div className="block-meta">
                                <Clock size={10} color="#94a3b8" />
                                <span>~{walkMins} min walk</span>
                                <span className="block-tag">Inside 1km</span>
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
            disabled={!wsConnected}
          />
          <button
            type="submit"
            className="send-btn"
            disabled={!wsConnected || !inputValue.trim()}
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
