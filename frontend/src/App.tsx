import { useState, useEffect, useRef } from 'react';
import { SplitScreenLayout } from './components/SplitScreenLayout';
import { AgentChat, ChatMessage, LogMessage } from './components/AgentChat';
import { InteractiveMap, MapViewport } from './components/InteractiveMap';

// One id per browser tab: the backend keeps this tab's conversation under it, so a reconnect does not lose a
// pending follow-up question. Clear starts a new conversation (the backend forgets it).
function tabSessionId(): string {
  try {
    const existing = sessionStorage.getItem('hdb-compass-session');
    if (existing) return existing;
    const id = crypto.randomUUID();
    sessionStorage.setItem('hdb-compass-session', id);
    return id;
  } catch {
    return crypto.randomUUID();   // storage blocked: memory still lasts for this connection
  }
}

const WS_URL = `ws://localhost:8000/ws?session=${encodeURIComponent(tabSessionId())}`;

// The chat opens on the mascot welcome (AgentChat) with no message; Clear returns to it.
const INITIAL_MESSAGES: ChatMessage[] = [];

export function App() {
  const [messages, setMessages] = useState<ChatMessage[]>(INITIAL_MESSAGES);
  const [bufferGeoJSON, setBufferGeoJSON] = useState<GeoJSON.Feature | null>(null);
  const [poisGeoJSON, setPoisGeoJSON] = useState<GeoJSON.FeatureCollection | null>(null);
  const [viewport, setViewport] = useState<MapViewport>({
    lat: 1.3508,
    lng: 103.8481,
    zoom: 13.5,
  });
  const [theme, setTheme] = useState<string>('Default');
  const [agentStatus, setAgentStatus] = useState<string>('READY');
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const wsRef = useRef<WebSocket | null>(null);
  const currentMsgIdRef = useRef<string | null>(null);

  // ---------------------------------------------------------------------------
  // WebSocket Lifecycle & Multi-Agent Streaming State Bridge
  // ---------------------------------------------------------------------------
  useEffect(() => {
    let reconnectTimeout: any = null;

    const connectWebSocket = () => {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsConnected(true);
        setAgentStatus('READY');
      };

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);

          switch (payload.type) {
            case 'CONNECTION_ESTABLISHED':
              setAgentStatus('READY');
              break;

            case 'AGENT_STARTED':
              setAgentStatus('SEARCHING');
              break;

            case 'STEP_UPDATE':
              setAgentStatus('SEARCHING');

              // 1. If buffer polygon received, render instantly into MapLibre
              if (payload.geojson_buffer) {
                setBufferGeoJSON(payload.geojson_buffer);
              }

              // 2. If POIs/blocks received, update map pins
              if (payload.geojson_amenities) {
                setPoisGeoJSON(payload.geojson_amenities);
              }

              // 3. Smoothly reposition camera
              if (payload.viewport) {
                setViewport(payload.viewport);
              }

              // 4. Attach step log to active message for expandable trace
              const activeId = currentMsgIdRef.current;
              if (activeId && payload.logs && payload.logs.length > 0) {
                const logItem = payload.logs[0];
                const newLog: LogMessage = {
                  id: `log-${Date.now()}-${Math.random().toString(36).substr(2, 4)}`,
                  type: 'agent',
                  title: logItem.step || payload.node || 'Processing',
                  detail: logItem.message,
                  timestamp: new Date().toLocaleTimeString(),
                  data: payload.geojson_buffer?.properties || payload.nearby_amenities,
                };

                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === activeId
                      ? {
                          ...m,
                          stepLogs: [...(m.stepLogs || []), newLog],
                        }
                      : m
                  )
                );
              }

              // 5. Final synthesized answer
              if (payload.summary && activeId) {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === activeId
                      ? {
                          ...m,
                          text: payload.summary,
                          status: 'completed',
                          structuredData: {
                            target_name: payload.summary.split('\n')[0].replace(/\*\*/g, ''),
                            total_blocks_found: payload.qualifying_blocks?.length || 0,
                            qualifying_blocks: payload.qualifying_blocks || [],
                            nearby_amenities: payload.nearby_amenities || [],
                            // Generic slots ready for other agents (Market SQL, Financial, Policy)
                            market_summary: payload.market_summary,
                            financial_summary: payload.financial_summary,
                            policy_notes: payload.policy_notes,
                          },
                        }
                      : m
                  )
                );
              }
              break;

            case 'AGENT_FINISHED':
              setAgentStatus('READY');
              if (currentMsgIdRef.current) {
                const finishedId = currentMsgIdRef.current;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === finishedId ? { ...m, status: 'completed' } : m
                  )
                );
                currentMsgIdRef.current = null;
              }
              break;

            case 'ERROR':
              setAgentStatus('ERROR');
              if (currentMsgIdRef.current) {
                const errorId = currentMsgIdRef.current;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === errorId
                      ? {
                          ...m,
                          text: `Unable to complete query: ${payload.error}`,
                          status: 'error',
                        }
                      : m
                  )
                );
                currentMsgIdRef.current = null;
              }
              break;

            default:
              break;
          }
        } catch (err) {
          console.error('[WS Parse Error]', err);
        }
      };

      ws.onclose = () => {
        setWsConnected(false);
        setAgentStatus('OFFLINE');
        reconnectTimeout = setTimeout(connectWebSocket, 2500);
      };

      ws.onerror = (err) => {
        console.error('[WS Socket Error]', err);
        ws.close();
      };
    };

    connectWebSocket();

    return () => {
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // ---------------------------------------------------------------------------
  // User Actions: Send Query & Clear
  // ---------------------------------------------------------------------------
  const handleSendQuery = (queryText: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      alert('Connecting to server. Please wait a moment...');
      return;
    }

    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // 1. User Message
    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      sender: 'user',
      text: queryText,
      timestamp: timeStr,
    };

    // 2. Pending Assistant Response
    const agentMsgId = `agent-${Date.now()}`;
    currentMsgIdRef.current = agentMsgId;

    const pendingAgentMsg: ChatMessage = {
      id: agentMsgId,
      sender: 'agent',
      text: 'Searching database and mapping results...',
      timestamp: timeStr,
      status: 'running',
      stepLogs: [],
    };

    setMessages((prev) => [...prev, userMsg, pendingAgentMsg]);

    // 3. Dispatch to multi-agent backend via WebSocket
    wsRef.current.send(
      JSON.stringify({
        type: 'RUN_AGENT',
        query: queryText,
        radius_meters: 1000.0,
      })
    );
  };

  const [selectedBlockHighlight, setSelectedBlockHighlight] = useState<string | null>(null);
  const [clearTrigger, setClearTrigger] = useState<number>(0);

  const handleClear = () => {
    // start a fresh conversation: the backend forgets the previous request and any open follow-up question
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ reset: true }));
    }
    setMessages(INITIAL_MESSAGES);
    setBufferGeoJSON(null);
    setPoisGeoJSON(null);
    setSelectedBlockHighlight(null);
    setClearTrigger((prev) => prev + 1);
    currentMsgIdRef.current = null;
    setAgentStatus('READY');
  };

  return (
    <SplitScreenLayout
      leftContent={
        <AgentChat
          messages={messages}
          agentStatus={agentStatus}
          wsConnected={wsConnected}
          onSendQuery={handleSendQuery}
          onClear={handleClear}
          onSelectBlock={(blockName) => setSelectedBlockHighlight(blockName)}
        />
      }
      rightContent={
        <InteractiveMap
          bufferGeoJSON={bufferGeoJSON}
          poisGeoJSON={poisGeoJSON}
          viewport={viewport}
          theme={theme}
          onThemeChange={setTheme}
          selectedBlockHighlight={selectedBlockHighlight}
          clearTrigger={clearTrigger}
        />
      }
    />
  );
}

export default App;
