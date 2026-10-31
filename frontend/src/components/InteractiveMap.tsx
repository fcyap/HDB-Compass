import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import * as maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
// @ts-ignore
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url';
import {
  Layers,
  Compass,
  Train,
  School,
  Utensils,
  ShoppingCart,
  X,
  Clock,
  Sparkles,
} from 'lucide-react';

// Explicitly register worker with MapLibre to avoid Vite worker isolation issues
if (typeof (maplibregl as any).setWorkerUrl === 'function') {
  (maplibregl as any).setWorkerUrl(workerUrl);
}

export interface MapViewport {
  lat: number;
  lng: number;
  zoom: number;
}

// Default radius of the pin circle and facility panel; chat questions about the pin use the same circle.
export const FACILITY_RADIUS_METERS = 500;
export const RADIUS_PRESETS: { meters: number; label: string }[] = [
  { meters: 200, label: '200 m' },
  { meters: 500, label: '500 m' },
  { meters: 1000, label: '1 km' },
  { meters: 1500, label: '1.5 km' },
  { meters: 2000, label: '2 km' },
];
const RADIUS_MIN = 100;
const RADIUS_MAX = 3000;
const RADIUS_STORAGE_KEY = 'hdb-compass-radius-m';

export const loadSavedRadius = (): number => {
  try {
    const v = Number(window.localStorage.getItem(RADIUS_STORAGE_KEY));
    if (Number.isFinite(v) && v >= RADIUS_MIN && v <= RADIUS_MAX) return v;
  } catch (e) { /* storage unavailable: use the default */ }
  return FACILITY_RADIUS_METERS;
};

// The pin's border: a 64-point circle computed in the browser (metres to degrees at the pin's latitude), so the
// border follows the slider instantly with no request per step. Singapore-sized radii make this accurate to <0.1%.
const circlePolygon = (lat: number, lng: number, radiusM: number): GeoJSON.Feature => {
  const dLat = radiusM / 111320;
  const dLng = radiusM / (111320 * Math.cos((lat * Math.PI) / 180));
  const ring: [number, number][] = [];
  for (let i = 0; i <= 64; i++) {
    const a = (2 * Math.PI * (i % 64)) / 64;
    ring.push([lng + dLng * Math.cos(a), lat + dLat * Math.sin(a)]);
  }
  return { type: 'Feature', properties: {}, geometry: { type: 'Polygon', coordinates: [ring] } };
};

interface InteractiveMapProps {
  bufferGeoJSON: GeoJSON.Feature | null;
  poisGeoJSON: GeoJSON.FeatureCollection | null;
  viewport: MapViewport;
  onMapClick?: (lat: number, lng: number) => void;
  theme?: string;
  onThemeChange?: (theme: string) => void;
  selectedBlockHighlight?: string | null;
  clearTrigger?: number;
  // A click on a station or town marker that came from a grouped-statistics answer asks that marker's drill-down
  // question (the marker carries the question text), exactly as its chip would.
  onDrillQuestion?: (question: string) => void;
  radiusMeters?: number;
  onRadiusChange?: (meters: number) => void;
}

export const ONEMAP_THEMES: Record<string, string> = {
  Night: 'https://www.onemap.gov.sg/maps/json/raster/mbstyle/Night.json',
  Default: 'https://www.onemap.gov.sg/maps/json/raster/mbstyle/Default.json',
  Grey: 'https://www.onemap.gov.sg/maps/json/raster/mbstyle/Grey.json',
  Original: 'https://www.onemap.gov.sg/maps/json/raster/mbstyle/Original.json',
};

type AmenityCategory = 'all' | 'mrt' | 'school' | 'hawker' | 'supermarket';

interface NearbyFacility {
  name: string;
  category: string;
  distance_meters: number;
  walking_mins: number;
  lat: number;
  lng: number;
  line_code?: string;
  rawCategory?: string;
  source?: string;
}

interface SelectedBlockInfo {
  name: string;
  lat: number;
  lng: number;
  price?: number;
  lease?: number;
  town?: string;
  isHdb?: boolean;
}

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
  selectedBlockHighlight,
  clearTrigger,
  onDrillQuestion,
  radiusMeters = FACILITY_RADIUS_METERS,
  onRadiusChange,
}) => {
  const onDrillQuestionRef = useRef(onDrillQuestion);
  onDrillQuestionRef.current = onDrillQuestion;
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const popupRef = useRef<maplibregl.Popup | null>(null);
  const centerMarkerRef = useRef<maplibregl.Marker | null>(null);
  const clickMarkerRef = useRef<maplibregl.Marker | null>(null);
  const facilityMarkerRef = useRef<maplibregl.Marker | null>(null);

  const [mapLoaded, setMapLoaded] = useState(false);
  // The clicked pin and its radius border ('pin-radius-source'), separate from the agent's buffer layers
  const [pinCenter, setPinCenter] = useState<{ lat: number; lng: number } | null>(null);
  const pinCenterRef = useRef(pinCenter);
  pinCenterRef.current = pinCenter;
  const radiusLabel = radiusMeters >= 1000 ? `${radiusMeters / 1000} km` : `${radiusMeters} m`;
  const radiusRef = useRef(radiusMeters);
  radiusRef.current = radiusMeters;
  const [cursorCoords, setCursorCoords] = useState<{ lat: number; lng: number } | null>(null);

  // PropertyGuru-style Facility Drawer state
  const [selectedCategory, setSelectedCategory] = useState<AmenityCategory>('all');
  const [selectedBlock, setSelectedBlock] = useState<SelectedBlockInfo | null>(null);
  const [allNearbyFacilities, setAllNearbyFacilities] = useState<NearbyFacility[]>([]);
  const [loadingFacilities, setLoadingFacilities] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);

  // Keep latest GeoJSON state accessible in theme switch callbacks
  const bufferGeoJSONRef = useRef(bufferGeoJSON);
  bufferGeoJSONRef.current = bufferGeoJSON;
  const poisGeoJSONRef = useRef(poisGeoJSON);
  poisGeoJSONRef.current = poisGeoJSON;

  // Compute category counts
  const categoryCounts = useMemo(() => {
    let mrt = 0, school = 0, hawker = 0, supermarket = 0;
    allNearbyFacilities.forEach((f) => {
      const cat = (f.category || '').toLowerCase();
      if (cat.includes('mrt') || cat.includes('rail') || cat.includes('transit')) mrt++;
      else if (cat.includes('school')) school++;
      else if (cat.includes('hawker') || cat.includes('food') || cat.includes('restaurant') || cat.includes('cafe')) hawker++;
      else if (cat.includes('supermarket') || cat.includes('mart') || cat.includes('grocery')) supermarket++;
    });
    return {
      all: allNearbyFacilities.length,
      mrt,
      school,
      hawker,
      supermarket,
    };
  }, [allNearbyFacilities]);

  // Filter facilities based on selected category pill
  const filteredFacilities = useMemo(() => {
    if (selectedCategory === 'all') return allNearbyFacilities;
    return allNearbyFacilities.filter((f) => {
      const cat = (f.category || '').toLowerCase();
      if (selectedCategory === 'mrt') return cat.includes('mrt') || cat.includes('rail') || cat.includes('transit');
      if (selectedCategory === 'school') return cat.includes('school');
      if (selectedCategory === 'hawker') return cat.includes('hawker') || cat.includes('food') || cat.includes('restaurant') || cat.includes('cafe');
      if (selectedCategory === 'supermarket') return cat.includes('supermarket') || cat.includes('mart') || cat.includes('grocery');
      return true;
    });
  }, [allNearbyFacilities, selectedCategory]);

  // Convert filtered facilities into GeoJSON FeatureCollection
  const facilitiesGeoJSON = useMemo<GeoJSON.FeatureCollection>(() => {
    return {
      type: 'FeatureCollection',
      features: filteredFacilities.map((f) => {
        let normCat = 'amenity';
        const c = (f.category || '').toLowerCase();
        if (c.includes('mrt') || c.includes('rail') || c.includes('transit')) normCat = 'mrt';
        else if (c.includes('school')) normCat = 'school';
        else if (c.includes('hawker') || c.includes('food') || c.includes('restaurant') || c.includes('cafe')) normCat = 'hawker';
        else if (c.includes('supermarket') || c.includes('mart') || c.includes('grocery')) normCat = 'supermarket';

        return {
          type: 'Feature',
          geometry: {
            type: 'Point',
            coordinates: [f.lng, f.lat],
          },
          properties: {
            name: f.name,
            category: normCat,
            rawCategory: f.rawCategory || f.category,
            distance_meters: f.distance_meters,
            walking_mins: f.walking_mins,
            source: f.source,
          },
        };
      }),
    };
  }, [filteredFacilities]);

  const facilitiesGeoJSONRef = useRef(facilitiesGeoJSON);
  facilitiesGeoJSONRef.current = facilitiesGeoJSON;

  // Fetch all nearby facilities around a coordinate
  const fetchFacilitiesForLocation = useCallback((lat: number, lng: number) => {
    setLoadingFacilities(true);
    fetch('/api/spatial/radius', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        lat,
        lng,
        radius_meters: radiusRef.current,
        include_buffer_polygon: false,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        setAllNearbyFacilities(data.results || []);
      })
      .catch((err) => {
        console.error('[Nearby Facilities Fetch Error]', err);
      })
      .finally(() => {
        setLoadingFacilities(false);
      });
  }, []);

  // Reset/Clear Map State whenever clearTrigger fires
  useEffect(() => {
    if (clearTrigger && clearTrigger > 0) {
      setSelectedBlock(null);
      setDrawerOpen(false);
      setAllNearbyFacilities([]);
      setPinCenter(null);
      if (clickMarkerRef.current) {
        clickMarkerRef.current.remove();
        clickMarkerRef.current = null;
      }
      if (centerMarkerRef.current) {
        centerMarkerRef.current.remove();
        centerMarkerRef.current = null;
      }
      if (facilityMarkerRef.current) {
        facilityMarkerRef.current.remove();
        facilityMarkerRef.current = null;
      }
      if (popupRef.current) {
        popupRef.current.remove();
        popupRef.current = null;
      }
      if (mapRef.current && mapLoaded) {
        const src1 = mapRef.current.getSource('agent-facilities-source') as maplibregl.GeoJSONSource;
        if (src1) src1.setData({ type: 'FeatureCollection', features: [] });
        const src2 = mapRef.current.getSource('agent-pois-source') as maplibregl.GeoJSONSource;
        if (src2) src2.setData({ type: 'FeatureCollection', features: [] });
        const src4 = mapRef.current.getSource('pin-radius-source') as maplibregl.GeoJSONSource;
        if (src4) src4.setData({ type: 'FeatureCollection', features: [] });
        const src3 = mapRef.current.getSource('agent-buffer-source') as maplibregl.GeoJSONSource;
        if (src3) src3.setData({ type: 'FeatureCollection', features: [] });
      }
    }
  }, [clearTrigger, mapLoaded]);

  // Update map facilities layer when filtered GeoJSON updates
  useEffect(() => {
    if (!mapRef.current || !mapLoaded) return;
    const src = mapRef.current.getSource('agent-facilities-source') as maplibregl.GeoJSONSource;
    if (src) {
      src.setData(facilitiesGeoJSON);
    }
  }, [facilitiesGeoJSON, mapLoaded]);

  // Handle block highlight selection from chat cards
  useEffect(() => {
    if (!selectedBlockHighlight || !poisGeoJSON?.features) return;
    const cleanHighlight = selectedBlockHighlight.toUpperCase();
    const match = poisGeoJSON.features.find((f: any) => {
      const name = (f.properties?.name || '').toUpperCase();
      return name.includes(cleanHighlight) || cleanHighlight.includes(name);
    });

    if (match && match.geometry && (match.geometry as any).coordinates) {
      const [lng, lat] = (match.geometry as any).coordinates;
      const props = match.properties || {};
      setSelectedBlock({
        name: props.name || selectedBlockHighlight,
        lat,
        lng,
        price: props.median_resale_price,
        lease: props.remaining_lease_years,
        town: props.town,
        isHdb: true,
      });
      setDrawerOpen(true);
      fetchFacilitiesForLocation(lat, lng);
      if (mapRef.current) {
        mapRef.current.flyTo({ center: [lng, lat], zoom: 15.5, speed: 1.2 });
      }
    }
  }, [selectedBlockHighlight, poisGeoJSON, fetchFacilitiesForLocation]);

  // Auto-select 1st candidate block when search results arrive
  useEffect(() => {
    if (!poisGeoJSON?.features || poisGeoJSON.features.length === 0) return;
    const candidates = poisGeoJSON.features.filter(
      (f: any) => f.properties?.category === 'hdb_candidate'
    );
    const target = candidates.length > 0 ? candidates[0] : poisGeoJSON.features[0];
    if (target && target.geometry && (target.geometry as any).coordinates) {
      const [lng, lat] = (target.geometry as any).coordinates;
      const props = target.properties || {};
      setSelectedBlock({
        name: props.name || 'Candidate Block',
        lat,
        lng,
        price: props.median_resale_price,
        lease: props.remaining_lease_years,
        town: props.town,
        isHdb: true,
      });
      setDrawerOpen(true);
      fetchFacilitiesForLocation(lat, lng);
    }
  }, [poisGeoJSON, fetchFacilitiesForLocation]);

  // Reusable helper to attach all custom analysis layers & interactions
  const setupLayers = useCallback((map: maplibregl.Map) => {
    // 0. Pin radius border (the circle around the user's clicked pin), under the agent's buffer
    const pinCircle = pinCenterRef.current
      ? wrapFeatureCollection(circlePolygon(pinCenterRef.current.lat, pinCenterRef.current.lng, radiusRef.current))
      : wrapFeatureCollection(null);
    if (!map.getSource('pin-radius-source')) {
      map.addSource('pin-radius-source', { type: 'geojson', data: pinCircle });
      map.addLayer({
        id: 'pin-radius-fill',
        type: 'fill',
        source: 'pin-radius-source',
        paint: { 'fill-color': '#008F8A', 'fill-opacity': 0.1 },
      });
      map.addLayer({
        id: 'pin-radius-line',
        type: 'line',
        source: 'pin-radius-source',
        paint: { 'line-color': '#008F8A', 'line-width': 2, 'line-dasharray': [2, 2] },
      });
    } else {
      (map.getSource('pin-radius-source') as maplibregl.GeoJSONSource).setData(pinCircle);
    }

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
          'fill-color': '#008F8A',
          'fill-opacity': 0.18,
        },
      });

      map.addLayer({
        id: 'agent-buffer-line',
        type: 'line',
        source: 'agent-buffer-source',
        paint: {
          'line-color': '#006865',
          'line-width': 2.5,
          'line-dasharray': [3, 1.5],
        },
      });
    } else {
      const src = map.getSource('agent-buffer-source') as maplibregl.GeoJSONSource;
      src.setData(wrapFeatureCollection(bufferGeoJSONRef.current));
    }

    // 2. Nearby Facilities Layer (Schools, Transit, Food & Dining, Groceries)
    if (!map.getSource('agent-facilities-source')) {
      map.addSource('agent-facilities-source', {
        type: 'geojson',
        data: facilitiesGeoJSONRef.current,
      });

      // Facility Glow Layer
      map.addLayer({
        id: 'agent-facilities-glow',
        type: 'circle',
        source: 'agent-facilities-source',
        paint: {
          'circle-radius': 11,
          'circle-color': [
            'match',
            ['get', 'category'],
            'mrt', '#38bdf8',
            'school', '#f59e0b',
            'hawker', '#7C3AED',
            'supermarket', '#10b981',
            '#a855f7',
          ],
          'circle-opacity': 0.38,
          'circle-blur': 0.6,
        },
      });

      // Facility Sharp Core Circle
      map.addLayer({
        id: 'agent-facilities-circle',
        type: 'circle',
        source: 'agent-facilities-source',
        paint: {
          'circle-radius': 5.5,
          'circle-color': [
            'match',
            ['get', 'category'],
            'mrt', '#0ea5e9',
            'school', '#f59e0b',
            'hawker', '#7C3AED',
            'supermarket', '#10b981',
            '#a855f7',
          ],
          'circle-stroke-width': 2,
          'circle-stroke-color': '#ffffff',
        },
      });

      // Facility Name Label
      map.addLayer({
        id: 'agent-facilities-label',
        type: 'symbol',
        source: 'agent-facilities-source',
        layout: {
          'text-field': ['get', 'name'],
          'text-size': 10,
          'text-offset': [0, 1.3],
          'text-anchor': 'top',
          'text-optional': true,
        },
        paint: {
          'text-color': '#073E40',
          'text-halo-color': '#FFFFFF',
          'text-halo-width': 2.2,
        },
      });

      // Interactive Click on Facility Pins
      map.on('click', 'agent-facilities-circle', (e: any) => {
        if (!e.features || !e.features[0]) return;
        const feature = e.features[0];
        const coordinates = (feature.geometry as any).coordinates.slice();
        const props = feature.properties || {};

        let catColor = '#a855f7';
        let catLabel = 'AMENITY';
        if (props.category === 'mrt') { catColor = '#0ea5e9'; catLabel = 'TRANSIT'; }
        else if (props.category === 'school') { catColor = '#f59e0b'; catLabel = 'PRIMARY SCHOOL'; }
        else if (props.category === 'hawker') { catColor = '#7C3AED'; catLabel = 'FOOD & DINING'; }
        else if (props.category === 'supermarket') { catColor = '#10b981'; catLabel = 'GROCERY'; }

        const html = `
          <div style="font-family: sans-serif; padding: 6px; color: #073E40; min-width: 190px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
              <span style="background: ${catColor}; color: #fff; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px; text-transform: uppercase;">
                ${catLabel}
              </span>
              <span style="font-size: 11px; font-weight: 700; color: #006865;">
                ${Math.round(props.distance_meters)}m
              </span>
            </div>
            <strong style="font-size: 13px; color: #073E40; display: block; margin-bottom: 4px;">
              ${props.name}
            </strong>
            <div style="font-size: 11px; color: #3F6B69; display: flex; align-items: center; gap: 4px;">
              🚶 ~${props.walking_mins} mins walk
            </div>
          </div>
        `;

        if (popupRef.current) popupRef.current.remove();
        popupRef.current = new maplibregl.Popup({ offset: 12 })
          .setLngLat(coordinates)
          .setHTML(html)
          .addTo(map);
      });

      map.on('mouseenter', 'agent-facilities-circle', () => {
        map.getCanvas().style.cursor = 'pointer';
      });
      map.on('mouseleave', 'agent-facilities-circle', () => {
        map.getCanvas().style.cursor = '';
      });
    } else {
      const src = map.getSource('agent-facilities-source') as maplibregl.GeoJSONSource;
      src.setData(facilitiesGeoJSONRef.current);
    }

    // 3. Candidate HDB Blocks Layer
    if (!map.getSource('agent-pois-source')) {
      map.addSource('agent-pois-source', {
        type: 'geojson',
        data: wrapFeatureCollection(poisGeoJSONRef.current),
      });

      // Green Candidate Block Glow
      map.addLayer({
        id: 'agent-pois-glow',
        type: 'circle',
        source: 'agent-pois-source',
        paint: {
          'circle-radius': 14,
          'circle-color': '#F36550',
          'circle-opacity': 0.45,
          'circle-blur': 0.6,
        },
      });

      // Green Sharp Core Dot
      map.addLayer({
        id: 'agent-pois-circle',
        type: 'circle',
        source: 'agent-pois-source',
        paint: {
          'circle-radius': 7,
          'circle-color': '#F36550',
          'circle-stroke-width': 2.5,
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
          'text-offset': [0, 1.4],
          'text-anchor': 'top',
          'text-optional': true,
        },
        paint: {
          'text-color': '#073E40',
          'text-halo-color': '#FFFFFF',
          'text-halo-width': 2.8,
        },
      });

      // Interactive Click Popup & Drawer Trigger on HDB Candidate Pins
      map.on('click', 'agent-pois-circle', (e: any) => {
        if (!e.features || !e.features[0]) return;
        const feature = e.features[0];
        const coordinates = (feature.geometry as any).coordinates.slice();
        const props = feature.properties || {};

        // A station or town marker from a grouped answer: show its figures and ask its drill-down question.
        if (props.drill_question) {
          const median = props.median_resale_price ? `$${Number(props.median_resale_price).toLocaleString()}` : '';
          if (popupRef.current) popupRef.current.remove();
          popupRef.current = new maplibregl.Popup({ offset: 12 })
            .setLngLat(coordinates)
            .setHTML(`<div style="font-family: sans-serif; padding: 6px; color: #073E40;"><strong>${props.name}</strong><br/>` +
              `<span style="font-size: 11px;">Median ${median} (${props.transaction_count} sales)</span><br/>` +
              `<span style="font-size: 11px; color: #006865;">Asking the next question...</span></div>`)
            .addTo(map);
          onDrillQuestionRef.current?.(String(props.drill_question));
          return;
        }

        const blockName = props.name || 'HDB Block';
        const priceStr = props.median_resale_price
          ? `$${Number(props.median_resale_price).toLocaleString()}`
          : 'Price on request';
        const leaseStr = props.remaining_lease_years
          ? `${Math.round(props.remaining_lease_years)} yrs lease`
          : '';

        // Select block, open PropertyGuru-style nearby drawer, and fetch facilities
        setSelectedBlock({
          name: blockName,
          lat: coordinates[1],
          lng: coordinates[0],
          price: props.median_resale_price,
          lease: props.remaining_lease_years,
          town: props.town,
          isHdb: true,
        });
        setDrawerOpen(true);
        fetchFacilitiesForLocation(coordinates[1], coordinates[0]);

        const html = `
          <div style="font-family: sans-serif; padding: 6px; color: #073E40; min-width: 210px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
              <span style="background: #F36550; color: #fff; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px;">
                HDB RESALE
              </span>
              <span style="font-size: 11px; font-weight: 700; color: #006865;">
                ${priceStr}
              </span>
            </div>
            <strong style="font-size: 13px; color: #073E40; display: block; margin-bottom: 4px;">
              ${blockName}
            </strong>
            <div style="font-size: 11px; color: #6E8F8C; margin-bottom: 6px;">
              ${leaseStr}
            </div>
            <div style="font-size: 11px; color: #006865; font-weight: 600; background: #E3F3F1; padding: 4px 6px; border-radius: 4px; display: flex; align-items: center; gap: 4px;">
              📍 Facilities Panel Opened
            </div>
          </div>
        `;

        if (popupRef.current) popupRef.current.remove();
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
  }, [fetchFacilitiesForLocation]);

  // Initialize MapLibre Map
  useEffect(() => {
    if (!mapContainerRef.current) return;

    const styleUrl = ONEMAP_THEMES[theme] || ONEMAP_THEMES.Default;

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: styleUrl,
      center: [viewport.lng, viewport.lat],
      zoom: viewport.zoom,
      minZoom: 10,
      maxZoom: 19,
      attributionControl: false,
    });

    // The chat card floats over the left of the map. Camera padding makes centring, flyTo and fitting use the
    // visible part of the map to the right of the card, so results never land underneath it.
    const applyPanelPadding = () => {
      const panel = document.querySelector('.left-panel');
      const left = panel ? Math.round(panel.getBoundingClientRect().right) : 0;
      map.setPadding({ left, top: 0, right: 0, bottom: 0 });
    };
    applyPanelPadding();
    window.addEventListener('resize', applyPanelPadding);
    map.once('remove', () => window.removeEventListener('resize', applyPanelPadding));

    map.on('load', () => {
      setMapLoaded(true);
      setupLayers(map);
      // Clean initial state: do NOT pre-load random amenities until user searches or clicks
    });

    map.on('mousemove', (e: any) => {
      if (e.lngLat) {
        setCursorCoords({
          lat: Number(e.lngLat.lat.toFixed(5)),
          lng: Number(e.lngLat.lng.toFixed(5)),
        });
      }
    });

    // Interactive Map Click: Click ANY block, street, or building on the map!
    map.on('click', async (e: any) => {
      if (!e.lngLat) return;
      const { lat, lng } = e.lngLat;

      // 1. Notify parent
      if (onMapClick) {
        onMapClick(lat, lng);
      }

      // 2. Reverse inspect the clicked point (detects nearest HDB block or coordinates)
      try {
        const res = await fetch(`/api/spatial/inspect_point?lat=${lat}&lng=${lng}`);
        const data = await res.json();

        const locName = data.name || `Pinned Location (${lat.toFixed(4)}° N, ${lng.toFixed(4)}° E)`;
        const targetLat = data.lat || lat;
        const targetLng = data.lng || lng;

        setSelectedBlock({
          name: locName,
          lat: targetLat,
          lng: targetLng,
          town: data.town,
          isHdb: data.is_hdb,
        });

        // 3. Drop Interactive Pin on clicked location
        if (clickMarkerRef.current) {
          clickMarkerRef.current.remove();
        }
        const el = document.createElement('div');
        el.className = 'target-center-marker';
        el.innerHTML = `
          <div class="target-pin-pulse" style="border-color: #073E40; background: rgba(7, 62, 64, 0.2);"></div>
          <div class="target-pin-core" style="background: #073E40; border-color: #ffffff; color: #F8EFDC; font-size: 11px; font-weight: 800; box-shadow: 0 0 14px #073E40;">📍</div>
          <div class="target-pin-label" style="background: #F8EFDC; border: 1px solid #073E40; color: #073E40;">${locName}</div>
        `;
        clickMarkerRef.current = new maplibregl.Marker({ element: el })
          .setLngLat([targetLng, targetLat])
          .addTo(map);

        setDrawerOpen(true);
        pinCenterRef.current = { lat: targetLat, lng: targetLng };
        setPinCenter({ lat: targetLat, lng: targetLng });
        fetchFacilitiesForLocation(targetLat, targetLng);
      } catch (err) {
        console.error('[Map Click Error]', err);
      }
    });

    mapRef.current = map;

    return () => {
      map.remove();
    };
  }, [onMapClick, setupLayers, fetchFacilitiesForLocation]);

  // Dynamically update basemap when user changes OneMap Theme in dropdown
  useEffect(() => {
    if (!mapRef.current || !mapLoaded) return;
    const map = mapRef.current;
    const styleUrl = ONEMAP_THEMES[theme] || ONEMAP_THEMES.Default;

    map.setStyle(styleUrl);
    map.once('style.load', () => {
      setupLayers(map);
    });
  }, [theme, mapLoaded, setupLayers]);

  // Redraw the pin's radius border when the pin or the radius changes
  useEffect(() => {
    if (!mapRef.current || !mapLoaded) return;
    const source = mapRef.current.getSource('pin-radius-source') as maplibregl.GeoJSONSource;
    if (source) {
      source.setData(wrapFeatureCollection(pinCenter ? circlePolygon(pinCenter.lat, pinCenter.lng, radiusMeters) : null));
    }
  }, [pinCenter, radiusMeters, mapLoaded]);

  // A new radius re-fetches the facilities around the pin (after a short pause while the slider is dragged)
  const firstRadiusRun = useRef(true);
  useEffect(() => {
    if (firstRadiusRun.current) {
      firstRadiusRun.current = false;
      return;
    }
    const pin = pinCenterRef.current;
    if (!pin) return;
    const t = window.setTimeout(() => fetchFacilitiesForLocation(pin.lat, pin.lng), 250);
    return () => window.clearTimeout(t);
  }, [radiusMeters, fetchFacilitiesForLocation]);

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

  // Handler when user clicks any of the top category pills
  const handlePillClick = (cat: AmenityCategory) => {
    setSelectedCategory(cat);
    setDrawerOpen(true);

    let refLat = selectedBlock?.lat;
    let refLng = selectedBlock?.lng;

    if (!refLat || !refLng) {
      if (bufferGeoJSON?.properties?.center_lat && bufferGeoJSON?.properties?.center_lng) {
        refLat = bufferGeoJSON.properties.center_lat;
        refLng = bufferGeoJSON.properties.center_lng;
        setSelectedBlock({
          name: bufferGeoJSON.properties.target_name || 'Target Location',
          lat: refLat,
          lng: refLng,
        });
      } else if (poisGeoJSON?.features && poisGeoJSON.features.length > 0) {
        const first = poisGeoJSON.features[0];
        const [l_lng, l_lat] = (first.geometry as any).coordinates;
        refLat = l_lat;
        refLng = l_lng;
        setSelectedBlock({
          name: first.properties?.name || 'Candidate Block',
          lat: refLat,
          lng: refLng,
          price: first.properties?.median_resale_price,
          lease: first.properties?.remaining_lease_years,
          town: first.properties?.town,
          isHdb: true,
        });
      } else {
        refLat = viewport.lat;
        refLng = viewport.lng;
        setSelectedBlock({
          name: 'Current Map Area',
          lat: refLat,
          lng: refLng,
        });
      }
    }

    if (refLat && refLng) {
      fetchFacilitiesForLocation(refLat, refLng);
    }
  };

  // Helper to fly to a clicked facility and show a marker
  const handleFacilityClick = (facility: NearbyFacility) => {
    if (!mapRef.current) return;
    mapRef.current.flyTo({
      center: [facility.lng, facility.lat],
      zoom: 16.5,
      speed: 1.3,
    });

    if (facilityMarkerRef.current) {
      facilityMarkerRef.current.remove();
    }

    const popup = new maplibregl.Popup({ offset: 12 })
      .setLngLat([facility.lng, facility.lat])
      .setHTML(`
        <div style="font-family: sans-serif; padding: 4px; color: #073E40;">
          <div style="font-size: 10px; font-weight: 700; color: #006865; text-transform: uppercase; margin-bottom: 2px;">
            ${facility.rawCategory || facility.category}
          </div>
          <strong style="font-size: 13px; color: #073E40; display: block; margin-bottom: 2px;">
            ${facility.name}
          </strong>
          <span style="font-size: 11px; color: #006865; font-weight: 600;">
            🚶 ${facility.walking_mins} mins walk (${Math.round(facility.distance_meters)}m)
          </span>
        </div>
      `)
      .addTo(mapRef.current);

    popupRef.current = popup;
  };

  // Facility category icons mapping
  const renderCategoryIcon = (cat: string) => {
    const c = (cat || '').toLowerCase();
    if (c.includes('mrt') || c.includes('rail') || c.includes('transit')) {
      return (
        <div className="pg-icon-circle pg-icon-mrt">
          <Train size={15} />
        </div>
      );
    }
    if (c.includes('school')) {
      return (
        <div className="pg-icon-circle pg-icon-school">
          <School size={15} />
        </div>
      );
    }
    if (c.includes('hawker') || c.includes('food') || c.includes('restaurant') || c.includes('cafe')) {
      return (
        <div className="pg-icon-circle pg-icon-hawker">
          <Utensils size={15} />
        </div>
      );
    }
    return (
      <div className="pg-icon-circle pg-icon-supermarket">
        <ShoppingCart size={15} />
      </div>
    );
  };

  // Intuitive, clean badge label generator
  const getBadgeLabel = (f: NearbyFacility) => {
    const cat = (f.category || '').toLowerCase();
    const raw = (f.rawCategory || '').toLowerCase();

    if (cat.includes('school')) {
      return f.distance_meters <= 1000 ? '★ MOE Priority 1 (≤ 1km)' : 'PRIMARY SCHOOL';
    }
    if (cat.includes('mrt') || cat.includes('transit')) {
      return 'TRANSIT';
    }
    if (cat.includes('supermarket') || cat.includes('grocery')) {
      return 'GROCERIES';
    }
    // Food & Dining category
    if (raw.includes('cafe')) return 'CAFE';
    if (raw.includes('bakery')) return 'BAKERY';
    if (raw.includes('restaurant')) return 'RESTAURANT';
    if (raw.includes('hawker')) return 'HAWKER';
    return 'FOOD & DINING';
  };

  return (
    <div className="map-canvas-container">
      <div ref={mapContainerRef} style={{ width: '100%', height: '100%' }} />

      {/* Radius control: presets and a slider for the pin's radius (also used by chat questions about the pin) */}
      <div className="radius-control" role="group" aria-label="Pin radius">
        <div className="radius-control-head">
          <span>Pin radius</span>
          <strong aria-live="polite">{radiusLabel}</strong>
        </div>
        <div className="radius-presets">
          {RADIUS_PRESETS.map((p) => (
            <button
              key={p.meters}
              type="button"
              className={`radius-preset ${radiusMeters === p.meters ? 'active' : ''}`}
              aria-label={`Set pin radius to ${p.label}`}
              aria-pressed={radiusMeters === p.meters}
              onClick={() => onRadiusChange?.(p.meters)}
            >
              {p.label}
            </button>
          ))}
        </div>
        <input
          type="range"
          className="radius-slider"
          min={RADIUS_MIN}
          max={RADIUS_MAX}
          step={50}
          value={radiusMeters}
          aria-label="Pin radius in metres"
          aria-valuetext={`${radiusMeters} metres`}
          onChange={(e) => onRadiusChange?.(Number(e.target.value))}
        />
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* PropertyGuru-Style Category Filter Pill Bar (Top Left)             */}
      {/* ------------------------------------------------------------------ */}
      <div className="pg-top-pill-bar">
        <button
          className={`pg-pill-btn ${selectedCategory === 'all' ? 'active' : ''}`}
          onClick={() => handlePillClick('all')}
        >
          <Sparkles size={13} />
          <span>All Amenities {allNearbyFacilities.length > 0 ? `(${categoryCounts.all})` : ''}</span>
        </button>

        <button
          className={`pg-pill-btn ${selectedCategory === 'mrt' ? 'active' : ''}`}
          onClick={() => handlePillClick('mrt')}
        >
          <Train size={13} />
          <span>Transit {allNearbyFacilities.length > 0 ? `(${categoryCounts.mrt})` : ''}</span>
        </button>

        <button
          className={`pg-pill-btn ${selectedCategory === 'school' ? 'active' : ''}`}
          onClick={() => handlePillClick('school')}
        >
          <School size={13} />
          <span>Schools {allNearbyFacilities.length > 0 ? `(${categoryCounts.school})` : ''}</span>
        </button>

        <button
          className={`pg-pill-btn ${selectedCategory === 'hawker' ? 'active' : ''}`}
          onClick={() => handlePillClick('hawker')}
        >
          <Utensils size={13} />
          <span>Food & Dining {allNearbyFacilities.length > 0 ? `(${categoryCounts.hawker})` : ''}</span>
        </button>

        <button
          className={`pg-pill-btn ${selectedCategory === 'supermarket' ? 'active' : ''}`}
          onClick={() => handlePillClick('supermarket')}
        >
          <ShoppingCart size={13} />
          <span>Groceries {allNearbyFacilities.length > 0 ? `(${categoryCounts.supermarket})` : ''}</span>
        </button>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* PropertyGuru-Style Nearby Facilities Drawer (Floating Left Panel) */}
      {/* ------------------------------------------------------------------ */}
      {drawerOpen && selectedBlock && (
        <div className="pg-facility-drawer">
          <div className="pg-drawer-header">
            <div className="pg-drawer-title-area">
              <span className="pg-block-badge">
                {selectedBlock.town ? `${selectedBlock.town} RESALE` : selectedBlock.isHdb ? 'HDB BLOCK' : 'PINNED LOCATION'}
              </span>
              <h3 className="pg-drawer-title">{selectedBlock.name}</h3>
              <p className="pg-drawer-meta">
                {selectedBlock.price
                  ? `Median $${Number(selectedBlock.price).toLocaleString()} · ${Math.round(selectedBlock.lease || 60)} yrs lease`
                  : `Proximity Scan (${radiusLabel} Radius)`}
              </p>
            </div>
            <button className="pg-close-btn" onClick={() => setDrawerOpen(false)} title="Close panel">
              <X size={14} />
            </button>
          </div>

          <div className="pg-facility-list">
            {loadingFacilities ? (
              <div className="pg-empty-state">
                <Clock size={18} style={{ animation: 'spin 1.5s linear infinite', margin: '0 auto 6px' }} />
                <span>Scanning {radiusLabel} radius around block...</span>
              </div>
            ) : filteredFacilities.length === 0 ? (
              <div className="pg-empty-state">
                <span>No facilities found in this category within {radiusLabel}.</span>
              </div>
            ) : (
              filteredFacilities.map((f, i) => (
                <div
                  key={`${f.name}-${i}`}
                  className="pg-facility-card"
                  onClick={() => handleFacilityClick(f)}
                >
                  <div className="pg-facility-left">
                    {renderCategoryIcon(f.category)}
                    <div className="pg-facility-info">
                      <div className="pg-facility-name" title={f.name}>
                        {f.name}
                      </div>
                      <div className="pg-facility-sub">
                        <span className={f.category.toLowerCase().includes('school') && f.distance_meters <= 1000 ? 'pg-priority-tag' : ''}>
                          {getBadgeLabel(f)}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="pg-facility-right">
                    <span className="pg-walk-time">
                      <Clock size={10} />
                      {f.walking_mins} min
                    </span>
                    <span className="pg-walk-dist">{Math.round(f.distance_meters)} m</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* Floating Basemap Theme Switcher */}
      <div className="map-floating-overlay">
        <div className="map-control-card">
          <Layers size={15} color="#008F8A" />
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
          <span className="legend-dot" style={{ background: '#073E40', boxShadow: '0 0 6px #073E40' }} />
          <span>Target Location</span>
        </div>
        <div className="legend-item">
          <span className="legend-line" style={{ borderBottom: '2px dashed #006865' }} />
          <span>Search Radius</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ background: '#F36550', boxShadow: '0 0 6px #F36550' }} />
          <span>Candidate Resale Blocks</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ background: '#0ea5e9', boxShadow: '0 0 6px #0ea5e9' }} />
          <span>Transit Stations</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ background: '#f59e0b', boxShadow: '0 0 6px #f59e0b' }} />
          <span>Primary Schools</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ background: '#7C3AED', boxShadow: '0 0 6px #7C3AED' }} />
          <span>Food & Dining</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ background: '#10b981', boxShadow: '0 0 6px #10b981' }} />
          <span>Groceries</span>
        </div>
      </div>

      {/* Coordinate Telemetry Indicator */}
      {cursorCoords && (
        <div className="coords-readout">
          <Compass size={12} color="#008F8A" />
          <span>
            {cursorCoords.lat.toFixed(5)}° N, {cursorCoords.lng.toFixed(5)}° E
          </span>
        </div>
      )}
    </div>
  );
};
