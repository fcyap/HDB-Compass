import { useState, useEffect, useRef, useCallback } from 'react';
import { SplitScreenLayout } from './components/SplitScreenLayout';
import { AgentChat, ChatMessage, LogMessage, ModelInfo } from './components/AgentChat';
import { InteractiveMap, MapViewport, FACILITY_RADIUS_METERS } from './components/InteractiveMap';

// One id per page load: the backend keeps this page's conversation under it, so a dropped WebSocket that reconnects
// does not lose a pending follow-up question. It is deliberately not stored: a reload (which empties the chat on
// screen) or a duplicated tab starts a new conversation, so the backend never applies a request the user can no
// longer see. Clear also starts a new conversation (the backend forgets it).
const SESSION_ID = crypto.randomUUID();
// Same origin as the page (the Vite dev/preview proxy forwards /ws to the backend), so it also works over HTTPS and
// through a tunnel, not only on localhost.
const websocketUrl = new URL('/ws', window.location.href);
websocketUrl.protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
websocketUrl.searchParams.set('session', SESSION_ID);
const WS_URL = websocketUrl.toString();

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
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const pendingResetRef = useRef(false);
  const currentMsgIdRef = useRef<string | null>(null);
  // The user's last map click; sent with each question so "here" / "this block" resolve to it.
  const pinnedRef = useRef<{ lat: number; lng: number } | null>(null);
  const handleMapClick = useCallback((lat: number, lng: number) => {
    pinnedRef.current = { lat, lng };
  }, []);

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
        fetch('/api/model/status')
          .then((response) => response.ok ? response.json() : Promise.reject())
          .then(setModelInfo)
          .catch(() => setModelInfo({ status: 'unavailable', message: 'Could not read model configuration.' }));
        if (pendingResetRef.current) {
          // Clear was pressed while disconnected: tell the backend now, before the next question
          ws.send(JSON.stringify({ reset: true }));
          pendingResetRef.current = false;
        }
      };

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);

          switch (payload.type) {
            case 'MODEL_STATUS':
              setModelInfo(payload);
              break;
            case 'CONNECTION_ESTABLISHED':
              setAgentStatus('READY');
              break;

            case 'AGENT_STARTED':
              setAgentStatus('SEARCHING');
              setBufferGeoJSON(null);
              setPoisGeoJSON(null);
              setSelectedBlockHighlight(null);
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
                          status: 'running',
                          structuredData: {
                            target_name: payload.summary.split('\n')[0].replace(/\*\*/g, ''),
                            total_blocks_found: payload.qualifying_blocks?.length || 0,
                            qualifying_blocks: payload.qualifying_blocks || [],
                            nearby_amenities: payload.nearby_amenities || [],
                            // Generic slots ready for other agents (Market SQL, Financial, Policy)
                            market_summary: payload.market_summary,
                            financial_summary: payload.financial_summary,
                            policy_notes: payload.policy_notes,
                            school_lookup: payload.school_lookup,
                            clarification: payload.clarification,
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
    if (agentStatus === 'SEARCHING') return;
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      alert('Connecting to server. Please wait a moment...');
      return;
    }

    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    setAgentStatus('SEARCHING');

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
        radius_meters: FACILITY_RADIUS_METERS,
        ...(pinnedRef.current ? { lat: pinnedRef.current.lat, lng: pinnedRef.current.lng } : {}),
      })
    );
  };

  const [selectedBlockHighlight, setSelectedBlockHighlight] = useState<string | null>(null);
  const [clearTrigger, setClearTrigger] = useState<number>(0);

  const handleClear = () => {
    // start a fresh conversation: the backend forgets the previous request and any open follow-up question.
    // While disconnected the reset waits for the reconnect (onopen), so it is never lost.
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ reset: true }));
    } else {
      pendingResetRef.current = true;
    }
    setMessages(INITIAL_MESSAGES);
    setBufferGeoJSON(null);
    setPoisGeoJSON(null);
    setSelectedBlockHighlight(null);
    setClearTrigger((prev) => prev + 1);
    pinnedRef.current = null;
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
          modelInfo={modelInfo}
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
          onMapClick={handleMapClick}
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
