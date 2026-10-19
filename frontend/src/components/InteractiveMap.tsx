import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
// @ts-ignore
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url';
import { Layers, Compass } from 'lucide-react';

// Explicitly register worker with MapLibre to avoid Vite worker isolation issues
if (typeof (maplibregl as any).setWorkerUrl === 'function') {
  (maplibregl as any).setWorkerUrl(workerUrl);
}

export interface MapViewport {
  lat: number;
  lng: number;
  zoom: number;
}

interface InteractiveMapProps {
  bufferGeoJSON: GeoJSON.Feature | null;
  poisGeoJSON: GeoJSON.FeatureCollection | null;
  viewport: MapViewport;
  onMapClick?: (lat: number, lng: number) => void;
  theme?: string;
  onThemeChange?: (theme: string) => void;
  selectedBlockHighlight?: string | null;
}

export const ONEMAP_THEMES: Record<string, string> = {
  Night: 'https://www.onemap.gov.sg/maps/json/raster/mbstyle/Night.json',
  Default: 'https://www.onemap.gov.sg/maps/json/raster/mbstyle/Default.json',
  Grey: 'https://www.onemap.gov.sg/maps/json/raster/mbstyle/Grey.json',
  Original: 'https://www.onemap.gov.sg/maps/json/raster/mbstyle/Original.json',
};

const wrapFeatureCollection = (geo: any): GeoJSON.FeatureCollection => {
  if (!geo) return { type: 'FeatureCollection', features: [] };
  if (geo.type === 'FeatureCollection') return geo;
  if (geo.type === 'Feature') return { type: 'FeatureCollection', features: [geo] };
  return { type: 'FeatureCollection', features: [] };
};

export const InteractiveMap: React.FC<InteractiveMapProps> = ({
  bufferGeoJSON,
  poisGeoJSON,
  viewport,
  onMapClick,
  theme = 'Night',
  onThemeChange,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const popupRef = useRef<maplibregl.Popup | null>(null);
  const centerMarkerRef = useRef<maplibregl.Marker | null>(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [cursorCoords, setCursorCoords] = useState<{ lat: number; lng: number } | null>(null);

  // Keep latest GeoJSON state accessible in theme switch callbacks
  const bufferGeoJSONRef = useRef(bufferGeoJSON);
  bufferGeoJSONRef.current = bufferGeoJSON;
  const poisGeoJSONRef = useRef(poisGeoJSON);
  poisGeoJSONRef.current = poisGeoJSON;

  // Reusable helper to attach all custom analysis layers & interactions
  const setupLayers = useCallback((map: maplibregl.Map) => {
    // 1. Buffer Polygon Layer
    if (!map.getSource('agent-buffer-source')) {
      map.addSource('agent-buffer-source', {
        type: 'geojson',
        data: wrapFeatureCollection(bufferGeoJSONRef.current),
      });

      map.addLayer({
        id: 'agent-buffer-fill',
        type: 'fill',
        source: 'agent-buffer-source',
        paint: {
          'fill-color': '#38bdf8',
          'fill-opacity': 0.22,
        },
      });

      map.addLayer({
        id: 'agent-buffer-line',
        type: 'line',
        source: 'agent-buffer-source',
        paint: {
          'line-color': '#0ea5e9',
          'line-width': 3,
          'line-dasharray': [3, 1.5],
        },
      });
    } else {
      const src = map.getSource('agent-buffer-source') as maplibregl.GeoJSONSource;
      src.setData(wrapFeatureCollection(bufferGeoJSONRef.current));
    }

    // 2. Candidate HDB Blocks & POIs Layer
    if (!map.getSource('agent-pois-source')) {
      map.addSource('agent-pois-source', {
        type: 'geojson',
        data: wrapFeatureCollection(poisGeoJSONRef.current),
      });

      // Glow circle
      map.addLayer({
        id: 'agent-pois-glow',
        type: 'circle',
        source: 'agent-pois-source',
        paint: {
          'circle-radius': 13,
          'circle-color': [
            'case',
            ['==', ['get', 'category'], 'hdb_candidate'],
            '#10b981',
            '#a855f7',
          ],
          'circle-opacity': 0.45,
          'circle-blur': 0.6,
        },
      });

      // Sharp Core Dot
      map.addLayer({
        id: 'agent-pois-circle',
        type: 'circle',
        source: 'agent-pois-source',
        paint: {
          'circle-radius': 6.5,
          'circle-color': [
            'case',
            ['==', ['get', 'category'], 'hdb_candidate'],
            '#10b981',
            '#c084fc',
          ],
          'circle-stroke-width': 2,
          'circle-stroke-color': '#ffffff',
        },
      });

      // Block Name Label
      map.addLayer({
        id: 'agent-pois-label',
        type: 'symbol',
        source: 'agent-pois-source',
        layout: {
          'text-field': ['get', 'name'],
          'text-size': 11,
          'text-offset': [0, 1.3],
          'text-anchor': 'top',
          'text-optional': true,
        },
        paint: {
          'text-color': '#f8fafc',
          'text-halo-color': '#090d16',
          'text-halo-width': 2.5,
        },
      });

      // Interactive Click Popup on HDB Candidate Pins
      map.on('click', 'agent-pois-circle', (e: any) => {
        if (!e.features || !e.features[0]) return;
        const feature = e.features[0];
        const coordinates = (feature.geometry as any).coordinates.slice();
        const props = feature.properties || {};

        const html = `
          <div style="font-family: sans-serif; padding: 4px; color: #0f172a; min-width: 200px;">
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
              <span style="background: #10b981; color: #fff; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px;">
                ${props.priority || 'HDB RESALE'}
              </span>
            </div>
            <strong style="font-size: 13px; color: #0f172a; display: block; margin-bottom: 4px;">
              ${props.name || 'HDB Block'}
            </strong>
            <div style="font-size: 11px; color: #059669; font-weight: 600; background: #ecfdf5; padding: 4px 6px; border-radius: 4px;">
              ✓ Within Search Radius
            </div>
          </div>
        `;

        if (popupRef.current) {
          popupRef.current.remove();
        }

        popupRef.current = new maplibregl.Popup({ offset: 12 })
          .setLngLat(coordinates)
          .setHTML(html)
          .addTo(map);
      });

      map.on('mouseenter', 'agent-pois-circle', () => {
        map.getCanvas().style.cursor = 'pointer';
      });
      map.on('mouseleave', 'agent-pois-circle', () => {
        map.getCanvas().style.cursor = '';
      });
    } else {
      const src = map.getSource('agent-pois-source') as maplibregl.GeoJSONSource;
      src.setData(wrapFeatureCollection(poisGeoJSONRef.current));
    }
  }, []);

  // Initialize MapLibre Map
  useEffect(() => {
    if (!mapContainerRef.current) return;

    const styleUrl = ONEMAP_THEMES[theme] || ONEMAP_THEMES.Night;

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: styleUrl,
      center: [viewport.lng, viewport.lat],
      zoom: viewport.zoom,
      minZoom: 10,
      maxZoom: 19,
      attributionControl: false,
    });

    map.addControl(
      new maplibregl.AttributionControl({
        compact: true,
        customAttribution: '© OneMap SG | SLA',
      }),
      'bottom-left'
    );

    map.addControl(new maplibregl.NavigationControl({ showCompass: true }), 'top-right');

    map.on('load', () => {
      setupLayers(map);
      setMapLoaded(true);
    });

    map.on('mousemove', (e: any) => {
      if (e.lngLat) {
        setCursorCoords({
          lat: Number(e.lngLat.lat.toFixed(5)),
          lng: Number(e.lngLat.lng.toFixed(5)),
        });
      }
    });

    map.on('click', (e: any) => {
      if (onMapClick && e.lngLat) {
        onMapClick(e.lngLat.lat, e.lngLat.lng);
      }
    });

    mapRef.current = map;

    return () => {
      map.remove();
    };
  }, []);

  // Dynamically update basemap when user changes OneMap Theme in dropdown
  useEffect(() => {
    if (!mapRef.current || !mapLoaded) return;
    const map = mapRef.current;
    const styleUrl = ONEMAP_THEMES[theme] || ONEMAP_THEMES.Night;

    // Apply new OneMap raster tile style
    map.setStyle(styleUrl);

    // Re-attach custom agent buffer & POI layers once new style has loaded
    map.once('style.load', () => {
      setupLayers(map);
    });
  }, [theme, mapLoaded, setupLayers]);

  // Update Buffer Polygon Layer
  useEffect(() => {
    if (!mapRef.current || !mapLoaded) return;
    const source = mapRef.current.getSource('agent-buffer-source') as maplibregl.GeoJSONSource;
    if (source) {
      source.setData(wrapFeatureCollection(bufferGeoJSON));
    }

    // Render prominent Gold Target Center Marker
    if (bufferGeoJSON && bufferGeoJSON.properties) {
      const { center_lat, center_lng, target_name } = bufferGeoJSON.properties;
      if (center_lat && center_lng) {
        if (centerMarkerRef.current) {
          centerMarkerRef.current.remove();
        }

        const el = document.createElement('div');
        el.className = 'target-center-marker';
        el.innerHTML = `
          <div class="target-pin-pulse"></div>
          <div class="target-pin-core">★</div>
          <div class="target-pin-label">${target_name || 'Target'}</div>
        `;

        centerMarkerRef.current = new maplibregl.Marker({ element: el })
          .setLngLat([center_lng, center_lat])
          .addTo(mapRef.current);
      }
    } else {
      if (centerMarkerRef.current) {
        centerMarkerRef.current.remove();
        centerMarkerRef.current = null;
      }
    }
  }, [bufferGeoJSON, mapLoaded]);

  // Update POIs & Blocks Markers Layer
  useEffect(() => {
    if (!mapRef.current || !mapLoaded) return;
    const source = mapRef.current.getSource('agent-pois-source') as maplibregl.GeoJSONSource;
    if (source) {
      source.setData(wrapFeatureCollection(poisGeoJSON));
    }
  }, [poisGeoJSON, mapLoaded]);

  // Camera flyTo on Viewport Change
  useEffect(() => {
    if (!mapRef.current || !mapLoaded) return;
    mapRef.current.flyTo({
      center: [viewport.lng, viewport.lat],
      zoom: viewport.zoom,
      speed: 1.2,
      curve: 1.4,
      essential: true,
    });
  }, [viewport, mapLoaded]);

  return (
    <div className="map-canvas-container">
      <div ref={mapContainerRef} style={{ width: '100%', height: '100%' }} />

      {/* Floating Basemap Theme Switcher */}
      <div className="map-floating-overlay">
        <div className="map-control-card">
          <Layers size={15} color="#38bdf8" />
          <select
            className="theme-select"
            value={theme}
            onChange={(e) => onThemeChange && onThemeChange(e.target.value)}
          >
            <option value="Night">OneMap Night (Dark)</option>
            <option value="Default">OneMap Default (Color)</option>
            <option value="Grey">OneMap Grey (Subtle)</option>
            <option value="Original">OneMap Original (Classic)</option>
          </select>
        </div>
      </div>

      {/* Clean Map Legend */}
      <div className="map-legend-card">
        <span className="legend-title">Map Legend</span>
        <div className="legend-item">
          <span className="legend-dot" style={{ background: '#f59e0b', boxShadow: '0 0 6px #f59e0b' }} />
          <span>Target Location</span>
        </div>
        <div className="legend-item">
          <span className="legend-line" style={{ borderBottom: '2px dashed #0ea5e9' }} />
          <span>Search Radius</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ background: '#10b981', boxShadow: '0 0 6px #10b981' }} />
          <span>Candidate Resale Blocks</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ background: '#a855f7', boxShadow: '0 0 6px #a855f7' }} />
          <span>Transit & Amenities</span>
        </div>
      </div>

      {/* Coordinate Telemetry Indicator */}
      {cursorCoords && (
        <div className="coords-readout">
          <Compass size={12} color="#38bdf8" />
          <span>
            {cursorCoords.lat.toFixed(5)}° N, {cursorCoords.lng.toFixed(5)}° E
          </span>
        </div>
      )}
    </div>
  );
};
