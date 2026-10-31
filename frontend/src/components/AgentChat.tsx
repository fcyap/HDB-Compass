import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Compass,
  ChevronDown,
  ChevronRight,
  School,
  Building,
  CheckCircle2,
  Clock,
  Zap,
  RotateCcw,
  Terminal,
  Sparkles,
  ArrowUp,
  MapPin,
} from 'lucide-react';
import { Mascot3D, MascotAvatar, useMascotFavicon } from './Mascot';
import { ShortlistTray } from './ShortlistTray';
import { HouseholdForm } from './HouseholdForm';
import { useShortlist, blockLabel } from '../shortlist';

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
  clarification?: { kind?: string; question?: string; needed?: string[]; options: { label: string; text: string }[] };
  school_lookup?: { status: string; schools: { name: string; distance_meters: number }[] };
  flat_type?: string;
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
  const [showPinHint, setShowPinHint] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const shortlist = useShortlist();
  const busy = agentStatus === 'SEARCHING';   // one answer at a time: the app tracks a single running reply

  // the composer grows with the text, up to about six lines, like Grok's
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 150)}px`;
  }, [inputValue]);

  useEffect(() => {
    if (!showPinHint) return;
    const t = setTimeout(() => setShowPinHint(false), 6000);
    return () => clearTimeout(t);
  }, [showPinHint]);
  const [expandedTraceIds, setExpandedTraceIds] = useState<Record<string, boolean>>({});
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll on new message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, agentStatus]);

  const toggleTrace = (id: string) => {
    setExpandedTraceIds((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!inputValue.trim() || !wsConnected || busy) return;
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
        {/* Before the first question, show the mascot welcome with suggested prompts */}
        {messages.length === 0 && (
          <div className="welcome-hero-card">
            <div className="welcome-mascot-stage">
              <Mascot3D size={150} mood="wave" />
            </div>
            <div className="welcome-header">
              <Sparkles size={16} color="#008F8A" />
              <span>Explore Singapore Public Housing</span>
            </div>
            <p className="welcome-desc">
              Hi, I'm your HDB-Compass home guide. Search flats by budget and distance, find primary schools near an
              address, see what is near a block, or check resale prices and market trends. Distances are approximate
              straight-line estimates, not official MOE registration distances.
            </p>
            <div className="welcome-prompts-grid">
              <button
                className="welcome-prompt-card"
                onClick={() => handleQuickPrompt('Find 4-room flats in Bishan under $950k within 1km of Ai Tong School')}
              >
                <div className="prompt-card-icon school">
                  <School size={14} />
                </div>
                <div className="prompt-card-text">
                  <span className="prompt-title">Ai Tong School</span>
                  <span className="prompt-subtitle">Bishan 4-Room &lt; $950k · ~1km straight-line</span>
                </div>
              </button>

              <button
                className="welcome-prompt-card"
                onClick={() => handleQuickPrompt('Find 4-room flats in Bishan under $950k within 1km of Catholic High School')}
              >
                <div className="prompt-card-icon school">
                  <School size={14} />
                </div>
                <div className="prompt-card-text">
                  <span className="prompt-title">Catholic High School</span>
                  <span className="prompt-subtitle">Bishan 4-Room &lt; $950k · ~1km straight-line</span>
                </div>
              </button>

              <button
                className="welcome-prompt-card"
                onClick={() => handleQuickPrompt('Find 4-room flats in Ang Mo Kio under $750k within 300m of an MRT')}
              >
                <div className="prompt-card-icon transit">
                  <Zap size={14} />
                </div>
                <div className="prompt-card-text">
                  <span className="prompt-title">Ang Mo Kio MRT</span>
                  <span className="prompt-subtitle">4-Room &lt; $750k · 300m MRT Walk</span>
                </div>
              </button>

              <button
                className="welcome-prompt-card"
                onClick={() => handleQuickPrompt('Find 4-room flats in Kallang under $800k within 500m of an MRT and hawker')}
              >
                <div className="prompt-card-icon location">
                  <Building size={14} />
                </div>
                <div className="prompt-card-text">
                  <span className="prompt-title">Kallang / Whampoa</span>
                  <span className="prompt-subtitle">4-Room &lt; $800k · 500m MRT &amp; Hawker</span>
                </div>
              </button>

              <button
                className="welcome-prompt-card"
                onClick={() => handleQuickPrompt('Which primary schools are near Blk 147 Bishan Street 11?')}
              >
                <div className="prompt-card-icon location">
                  <MapPin size={14} />
                </div>
                <div className="prompt-card-text">
                  <span className="prompt-title">Schools near an address</span>
                  <span className="prompt-subtitle">Blk 147 Bishan St 11 · within 2km</span>
                </div>
              </button>

              <button
                className="welcome-prompt-card"
                onClick={() => handleQuickPrompt('How is the HDB resale market doing now?')}
              >
                <div className="prompt-card-icon transit">
                  <Compass size={14} />
                </div>
                <div className="prompt-card-text">
                  <span className="prompt-title">Resale market now</span>
                  <span className="prompt-subtitle">Latest medians &amp; recent change</span>
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
                    <MascotAvatar size={24} fallback={<Compass size={14} color="#008F8A" />} />
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
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{ a: (props) => <a {...props} target="_blank" rel="noopener noreferrer" /> }}
                    >
                      {msg.text}
                    </ReactMarkdown>
                  </div>

                  {/* Option chips for an open follow-up question (latest answer only): a click sends the option's text */}
                  {msg.id === messages[messages.length - 1]?.id && (msg.structuredData?.clarification?.options?.length ?? 0) > 0 && (
                    <div className="option-chips">
                      {msg.structuredData!.clarification!.options.map((o) => (
                        <button
                          key={o.label}
                          type="button"
                          className="option-chip"
                          disabled={agentStatus !== 'READY'}
                          onClick={() => handleQuickPrompt(o.text)}
                        >
                          {o.label}
                        </button>
                      ))}
                    </div>
                  )}

                  {/* Household facts form for an open financial follow-up (latest answer only) */}
                  {msg.id === messages[messages.length - 1]?.id && msg.structuredData?.clarification?.kind === 'financial' &&
                    (msg.structuredData.clarification.needed?.length ?? 0) > 0 && (
                    <HouseholdForm
                      needed={msg.structuredData.clarification.needed!}
                      disabled={agentStatus !== 'READY'}
                      onSubmit={handleQuickPrompt}
                    />
                  )}

                  {/* Candidate HDB Blocks Cards */}
                  {msg.structuredData?.qualifying_blocks && msg.structuredData.qualifying_blocks.length > 0 && (
                    <div className="candidate-blocks-section">
                      <div className="section-title-bar">
                        <span className="section-title">
                          <CheckCircle2 size={13} color="#008F8A" />
                          Matching HDB Resale Blocks ({msg.structuredData.total_blocks_found || msg.structuredData.qualifying_blocks.length}):
                        </span>
                        <span className="section-subtext">Click any pin on map</span>
                      </div>

                      <div className="candidate-cards-grid">
                        {msg.structuredData.qualifying_blocks.slice(0, 6).map((blockStr, bIdx) => {
                          const distMatch = blockStr.match(/\((\d+(?:\.\d+)?)m\)/);
                          let dist = distMatch ? `${distMatch[1]}m` : null;
                          const cleanName = blockStr.replace(/\s*\(\d+(?:\.\d+)?m\)/, '').trim();

                          // If distance not in block string, cross-reference response text for block stats
                          let priceInfo: string | null = null;
                          let leaseInfo: string | null = null;
                          if (msg.text) {
                            const blockLine = msg.text.split('\n').find((l) =>
                              l.toUpperCase().includes(cleanName.toUpperCase())
                            );
                            if (blockLine) {
                              if (!dist) {
                                const lineDist = blockLine.match(/(\d+(?:\.\d+)?)\s*m\b/);
                                if (lineDist) dist = `${lineDist[1]}m`;
                              }
                              const linePrice = blockLine.match(/median\s+\$([0-9,]+)/i);
                              if (linePrice) priceInfo = `$${linePrice[1]}`;
                              const lineLease = blockLine.match(/(\d+)\s*(?:yrs?|years?)\s*lease/i);
                              if (lineLease) leaseInfo = `${lineLease[1]} yrs`;
                            }
                          }

                          const flatType = msg.structuredData?.flat_type || '4 ROOM';
                          const blockName = blockLabel(cleanName);
                          const saved = shortlist.has({ block: blockName, flat_type: flatType });

                          return (
                            <div
                              key={bIdx}
                              className="candidate-block-pill"
                              onClick={() => onSelectBlock && onSelectBlock(cleanName)}
                            >
                              <div className="block-pill-header">
                                <span className="block-code" title={cleanName}>{cleanName}</span>
                                {dist && <span className="block-distance-badge">{dist}</span>}
                              </div>
                              <div className="card-actions">
                                <button type="button" className="card-action" disabled={agentStatus !== 'READY'}
                                  onClick={(e) => { e.stopPropagation(); handleQuickPrompt(`Is the price fair for a ${flatType} at ${blockName}?`); }}>
                                  Price check
                                </button>
                                <button type="button" className="card-action" disabled={agentStatus !== 'READY'}
                                  onClick={(e) => { e.stopPropagation(); handleQuickPrompt(`Can we afford a ${flatType} at ${blockName}?`); }}>
                                  Afford?
                                </button>
                                <button type="button" className={`card-action card-star${saved ? ' saved' : ''}`}
                                  aria-pressed={saved} aria-label={saved ? `Remove ${blockName} from shortlist` : `Add ${blockName} to shortlist`}
                                  title={saved ? 'Remove from shortlist' : 'Add to shortlist'}
                                  onClick={(e) => { e.stopPropagation(); shortlist.toggle({ block: blockName, flat_type: flatType }); }}>
                                  {saved ? '\u2605' : '\u2606'}
                                </button>
                              </div>
                              <div className="block-meta">
                                {dist ? (
                                  <>
                                    <Clock size={10} color="#6E8F8C" />
                                    {/* straight-line metres at ~80 m a minute; not a routed walk */}
                                    <span>~{Math.max(1, Math.round(parseFloat(dist) / 80))} min walk (straight-line)</span>
                                  </>
                                ) : priceInfo ? (
                                  <span>{priceInfo}{leaseInfo ? ` • ${leaseInfo} lease` : ''}</span>
                                ) : (
                                  <span>HDB Resale Flat</span>
                                )}
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
                      <Zap size={12} color="#F36550" />
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
                          <Terminal size={12} color="#6E8F8C" />
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

      {/* Composer, after Grok's: one floating pill with a helper button on the left and a round send button on the
          right. Only controls that work here: no mic or model picker. */}
      <footer className="chat-input-bar">
        {showPinHint && (
          <div className="composer-hint" role="status">
            <MapPin size={13} /> Click anywhere on the map to pin a spot and see what's nearby.
          </div>
        )}
        <ShortlistTray
          items={shortlist.items}
          disabled={agentStatus !== 'READY'}
          onRemove={shortlist.remove}
          onClear={shortlist.clear}
          onCompare={handleQuickPrompt}
        />
        <form onSubmit={handleSubmit} className={`composer${wsConnected ? '' : ' offline'}`}>
          <button
            type="button"
            className="composer-icon-btn"
            onClick={() => setShowPinHint((v) => !v)}
            aria-label="How to pin a location on the map"
            title="Pin a location on the map"
          >
            <MapPin size={17} />
          </button>
          <textarea
            ref={inputRef}
            rows={1}
            className="composer-input"
            placeholder={wsConnected ? 'Ask anything about HDB flats…' : 'Connecting to HDB-Compass…'}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => {
              // Enter sends, Shift+Enter adds a line; never while an input method is composing text
              if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                handleSubmit();
              }
            }}
            disabled={!wsConnected}
            aria-label="Message HDB-Compass"
          />
          <button
            type="submit"
            className={`composer-send${busy ? ' busy' : ''}`}
            disabled={!wsConnected || busy || !inputValue.trim()}
            aria-label={busy ? 'Working on your question' : 'Send'}
            title={busy ? 'Working on your question…' : 'Send (Enter)'}
          >
            {busy ? <span className="composer-spinner" /> : <ArrowUp size={18} strokeWidth={2.4} />}
          </button>
        </form>
      </footer>
    </aside>
  );
};
