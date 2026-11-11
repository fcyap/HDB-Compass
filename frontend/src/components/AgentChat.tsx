import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Components } from 'react-markdown';

// A link the answer wrote as a bare URL ("... remains unchanged https://www.hdb.gov.sg/...") is shown as a short
// source label, "(hdb.gov.sg)", with the full address on hover, instead of a long address run into the sentence.
// Links with their own text are left as they are. Every link opens in a new tab.
function SourceLink({ href, children, node: _node, ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement> & { node?: unknown }) {
  const text = Array.isArray(children) ? children.join('') : String(children ?? '');
  let label: React.ReactNode = children;
  if (href && /^https?:\/\//.test(text.trim())) {
    try {
      label = `(${new URL(href).hostname.replace(/^www\./, '')})`;
    } catch {
      label = '(source)';
    }
  }
  return <a {...props} href={href} title={href} target="_blank" rel="noopener noreferrer">{label}</a>;
}
const MARKDOWN_COMPONENTS: Components = { a: SourceLink };
import {
  ChevronDown,
  ChevronRight,
  School,
  Building,
  Clock,
  Zap,
  SquarePen,
  ArrowUp,
  MapPin,
  TrendingUp,
  TrainFront,
  Star,
} from 'lucide-react';
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
  // conversational style (HDB_ANSWER_STYLE): each card's own figures, so they are not read back out of the answer text
  candidate_cards?: { name: string; price?: number | null; lease?: number | null; distance?: number | null }[];
}

// A composed reply (conversational style) keeps code's checked facts under this line; they show as collapsed Details
const DETAILS_SEP = '\n\n---\n\n**Details (checked figures)**\n\n';

export interface ChatMessage {
  id: string;
  sender: 'user' | 'agent';
  text: string;
  timestamp: string;
  status?: 'running' | 'completed' | 'error';
  structuredData?: StructuredAgentData;
  stepLogs?: LogMessage[];
  streamText?: string;   // the composed reply as it streams in (REPLY_DELTA), until the final answer replaces it
}

export interface ModelInfo {
  profile?: string;
  model?: string;
  mode?: string;
  status?: string;
  message?: string;
  answer_style?: string;   // "conversational" on the Claude app; every other value renders as before
}

export interface ModelOption {
  id: string;
  label: string;
  available: boolean;
  reason?: string;
}

interface AgentChatProps {
  messages: ChatMessage[];
  agentStatus: string;
  wsConnected: boolean;
  modelInfo?: ModelInfo;
  modelOptions?: ModelOption[];
  selectedModel?: string;
  onSelectModel?: (id: string) => void;
  onSendQuery: (query: string) => void;
  onClear: () => void;
  onSelectBlock?: (blockSummary: string) => void;
}

// Suggested first questions on the empty chat; a click sends `query` as typed
const SUGGESTIONS: { icon: React.ElementType; title: string; subtitle: string; query: string }[] = [
  { icon: School, title: 'Ai Tong School', subtitle: 'Bishan 4-room under $950k, ~1 km', query: 'Find 4-room flats in Bishan under $950k within 1km of Ai Tong School' },
  { icon: School, title: 'Catholic High School', subtitle: 'Bishan 4-room under $950k, ~1 km', query: 'Find 4-room flats in Bishan under $950k within 1km of Catholic High School' },
  { icon: TrainFront, title: 'Ang Mo Kio MRT', subtitle: '4-room under $750k, 300 m to MRT', query: 'Find 4-room flats in Ang Mo Kio under $750k within 300m of an MRT' },
  { icon: Building, title: 'Kallang / Whampoa', subtitle: '4-room under $800k, MRT & hawker', query: 'Find 4-room flats in Kallang under $800k within 500m of an MRT and hawker' },
  { icon: MapPin, title: 'Schools near an address', subtitle: 'Blk 147 Bishan St 11, within 2 km', query: 'Which primary schools are near Blk 147 Bishan Street 11?' },
  { icon: TrendingUp, title: 'Resale market now', subtitle: 'Latest medians & recent change', query: 'How is the HDB resale market doing now?' },
];

/** The HDB-Compass mark: a compass needle on a teal gradient tile (same art as public/compass-mark.svg). */
const BrandMark: React.FC<{ size?: number }> = ({ size = 28 }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" className="brand-mark">
    <defs>
      <linearGradient id="bm-bg" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stopColor="#00A39D" />
        <stop offset="0.55" stopColor="#006865" />
        <stop offset="1" stopColor="#073E40" />
      </linearGradient>
      <linearGradient id="bm-tip" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stopColor="#FFAB94" />
        <stop offset="1" stopColor="#F36550" />
      </linearGradient>
    </defs>
    <rect width="32" height="32" rx="9" fill="url(#bm-bg)" />
    <path d="M16 6.5 19.2 16H12.8Z" fill="url(#bm-tip)" />
    <path d="M12.8 16H19.2L16 25.5Z" fill="#F8EFDC" />
    <circle cx="16" cy="16" r="1.6" fill="#073E40" />
  </svg>
);

export const AgentChat: React.FC<AgentChatProps> = ({
  messages,
  agentStatus,
  wsConnected,
  modelInfo,
  modelOptions = [],
  selectedModel,
  onSelectModel,
  onSendQuery,
  onClear,
  onSelectBlock,
}) => {
  const [inputValue, setInputValue] = useState('');
  const [showPinHint, setShowPinHint] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const shortlist = useShortlist();
  const busy = agentStatus === 'SEARCHING';   // one answer at a time: the app tracks a single running reply

  // the composer grows with the text, up to about six lines
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

  const modelLine = !modelInfo ? 'Reading model configuration…' : modelInfo.status === 'fallback'
    ? `Model fallback: ${modelInfo.message}`
    : modelInfo.mode === 'deterministic'
      ? 'Rule-based mode — no language model. Use explicit search criteria.'
      : modelInfo.status === 'error' || modelInfo.status === 'unavailable'
        ? `Model unavailable: ${modelInfo.message}`
        : `Model configured: ${modelInfo.profile} / ${modelInfo.model}`;
  const modelWarning = !!modelInfo && ['fallback', 'error', 'unavailable'].includes(modelInfo.status || '');

  return (
    <aside className="left-panel">
      <header className="header-bar">
        <div className="brand-badge">
          <BrandMark size={28} />
          <h1 className="brand-title">HDB-Compass</h1>
        </div>

        <div className="header-actions">
          {modelOptions.length > 0 && onSelectModel && (
            <select
              className="model-select"
              aria-label="Model"
              title="The model that answers your next question"
              value={selectedModel}
              disabled={busy}
              onChange={(e) => onSelectModel(e.target.value)}
            >
              {modelOptions.map((o) => (
                <option key={o.id} value={o.id} disabled={!o.available} title={o.available ? '' : o.reason}>
                  {o.label}{o.available ? '' : ' (unavailable)'}
                </option>
              ))}
            </select>
          )}
          <span
            className={`status-dot ${wsConnected ? 'connected' : 'disconnected'}`}
            role="status"
            aria-label={wsConnected ? 'Connected' : 'Connecting'}
            title={wsConnected ? 'Connected' : 'Connecting…'}
          />
          <button onClick={onClear} className="clear-btn" title="New chat" aria-label="New chat">
            <SquarePen size={16} />
          </button>
        </div>
      </header>

      <div className="chat-container">
        {/* Before the first question: a greeting and suggested prompts */}
        {messages.length === 0 && (
          <div className="welcome-hero-card">
            <h2 className="welcome-title">Where would you like to call home?</h2>
            <p className="welcome-desc">
              Search flats by budget and distance, find schools near an address, or check resale prices and trends.
            </p>
            <div className="welcome-prompts-grid">
              {SUGGESTIONS.map(({ icon: Icon, title, subtitle, query }) => (
                <button key={title} className="welcome-prompt-card" onClick={() => handleQuickPrompt(query)}>
                  <Icon size={15} className="prompt-card-icon" />
                  <span className="prompt-card-text">
                    <span className="prompt-title">{title}</span>
                    <span className="prompt-subtitle">{subtitle}</span>
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Messages */}
        {messages.map((msg) => (
          <div key={msg.id} className={`chat-message-row ${msg.sender}`}>
            {msg.sender === 'user' ? (
              <div className="user-message-bubble" title={msg.timestamp}>
                <p className="message-text">{msg.text}</p>
              </div>
            ) : (
              <div className="agent-message-card">
                {/* While no answer has arrived, a shimmering line names the current step */}
                {msg.status === 'running' && !msg.structuredData && msg.streamText ? (
                  <div className="agent-card-body">
                    <div className="agent-summary-text streaming">
                      <ReactMarkdown remarkPlugins={[remarkGfm]} components={MARKDOWN_COMPONENTS}>{msg.streamText}</ReactMarkdown>
                      <span className="stream-caret" aria-hidden="true" />
                    </div>
                  </div>
                ) : msg.status === 'running' && !msg.structuredData ? (
                  <div className="agent-running-banner" role="status">
                    <span className="thinking-orb" />
                    <span className="thinking-text">
                      {msg.stepLogs?.length ? msg.stepLogs[msg.stepLogs.length - 1].title : 'Thinking'}
                    </span>
                  </div>
                ) : (
                <div className="agent-card-body">
                  <div className="agent-summary-text">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={MARKDOWN_COMPONENTS}
                    >
                      {msg.text.includes(DETAILS_SEP) ? msg.text.split(DETAILS_SEP)[0] : msg.text}
                    </ReactMarkdown>
                  </div>
                  {msg.text.includes(DETAILS_SEP) && (
                    <details className="answer-details">
                      <summary>Details (checked figures)</summary>
                      <div className="agent-summary-text">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={MARKDOWN_COMPONENTS}
                        >
                          {msg.text.split(DETAILS_SEP).slice(1).join(DETAILS_SEP)}
                        </ReactMarkdown>
                      </div>
                    </details>
                  )}

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
                          Matching blocks
                          <span className="section-count">
                            {msg.structuredData.total_blocks_found || msg.structuredData.qualifying_blocks.length}
                          </span>
                        </span>
                        <span className="section-subtext">Tap a card to find it on the map</span>
                      </div>

                      <div className="candidate-cards-grid">
                        {msg.structuredData.qualifying_blocks.slice(0, 6).map((blockStr, bIdx) => {
                          const distMatch = blockStr.match(/\((\d+(?:\.\d+)?)m\)/);
                          let dist = distMatch ? `${distMatch[1]}m` : null;
                          const cleanName = blockStr.replace(/\s*\(\d+(?:\.\d+)?m\)/, '').trim();

                          // If distance not in block string, cross-reference response text for block stats
                          let priceInfo: string | null = null;
                          let leaseInfo: string | null = null;
                          const card = msg.structuredData?.candidate_cards?.find(
                            (c) => c.name.toUpperCase() === cleanName.toUpperCase());
                          if (card) {   // the card's own figures (conversational style)
                            if (!dist && card.distance != null) dist = `${card.distance}m`;
                            if (card.price != null) priceInfo = `$${Math.round(card.price).toLocaleString('en-US')}`;
                            if (card.lease != null) leaseInfo = `${Math.round(card.lease)} yrs`;
                          } else if (msg.text) {
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
                                <button type="button" className={`card-star${saved ? ' saved' : ''}`}
                                  aria-pressed={saved} aria-label={saved ? `Remove ${blockName} from shortlist` : `Add ${blockName} to shortlist`}
                                  title={saved ? 'Remove from shortlist' : 'Add to shortlist'}
                                  onClick={(e) => { e.stopPropagation(); shortlist.toggle({ block: blockName, flat_type: flatType }); }}>
                                  <Star size={14} fill={saved ? 'currentColor' : 'none'} />
                                </button>
                              </div>
                              <div className="block-meta">
                                {dist ? (
                                  <>
                                    <Clock size={11} />
                                    {/* straight-line metres at ~80 m a minute; not a routed walk */}
                                    <span>{dist} · ~{Math.max(1, Math.round(parseFloat(dist) / 80))} min walk</span>
                                  </>
                                ) : priceInfo ? (
                                  <span>{priceInfo}{leaseInfo ? ` · ${leaseInfo} lease` : ''}</span>
                                ) : (
                                  <span>HDB resale flat</span>
                                )}
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
                      <Zap size={12} />
                      <span>
                        Nearest transit: <strong>{msg.structuredData.nearby_amenities[0].name}</strong> ·{' '}
                        <strong>{msg.structuredData.nearby_amenities[0].distance_meters?.toFixed(0)} m</strong>
                      </span>
                    </div>
                  )}
                </div>
                )}

                {/* Collapsible execution trace */}
                {msg.stepLogs && msg.stepLogs.length > 0 && (
                  <div className="trace-accordion">
                    <button className="trace-toggle-btn" onClick={() => toggleTrace(msg.id)} aria-expanded={!!expandedTraceIds[msg.id]}>
                      <span>{modelInfo?.answer_style === 'conversational' ? 'How I worked this out · ' : ''}{msg.stepLogs.length} {msg.stepLogs.length === 1 ? 'step' : 'steps'}</span>
                      {expandedTraceIds[msg.id] ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
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
            )}
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      {/* Composer, after ChatGPT's: one floating pill with a helper button on the left and a round send button on the
          right, and a one-line note underneath. Only controls that work here: no mic or model picker. */}
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
            placeholder={wsConnected ? 'Ask anything about HDB flats' : 'Connecting to HDB-Compass…'}
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
        <p className={`composer-footnote${modelWarning ? ' warn' : ''}`} role="status">
          {modelLine} · Distances are straight-line estimates.
        </p>
      </footer>
    </aside>
  );
};
