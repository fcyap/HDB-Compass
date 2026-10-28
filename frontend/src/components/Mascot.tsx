import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

/**
 * HDB-Compass mascot: a little house holding a compass (public/mascot/hdb-compass-home-guide.glb).
 *
 * The model's parts are flat at the origin with their positions baked in, so the wave and the compass needle are
 * animated by re-parenting those parts onto pivots placed at the shoulder and the compass centre.
 *
 * Browsers allow only a few live WebGL canvases per page, so:
 *  - <Mascot3D>     a live, animated canvas: use it in one or two places (welcome card, "thinking" banner);
 *  - <MascotAvatar> a still image rendered once and shared, for the logo and every chat avatar.
 */

const MODEL_URL = '/mascot/hdb-compass-home-guide.glb';
const SHOULDER = new THREE.Vector3(0.68, 1.22, 0.05);     // right shoulder, from the arm's bounding box
const COMPASS_CENTRE = new THREE.Vector3(-1.03, 1.08, 0.58);
const WAVE_PARTS = ['Right arm', 'Waving hand', 'Thumb'];
const NEEDLE_PARTS = ['Compass needle coral', 'Compass needle teal'];

export type MascotFraming = 'full' | 'portrait';
export type MascotMood = 'wave' | 'thinking' | 'idle';

let modelPromise: Promise<THREE.Group> | null = null;
function loadModel(): Promise<THREE.Group> {
  if (!modelPromise) {
    modelPromise = new GLTFLoader().loadAsync(MODEL_URL).then((gltf) => gltf.scene);
  }
  return modelPromise;
}

// GLTFLoader replaces spaces in node names (and suffixes duplicates), so match on the original name
function findParts(root: THREE.Object3D, names: string[]): THREE.Object3D[] {
  const wanted = new Set(names);
  const found: THREE.Object3D[] = [];
  root.traverse((o) => {
    if (wanted.has(o.name.replace(/_/g, ' ').replace(/ \d+$/, ''))) found.push(o);
  });
  return found;
}

function pivotAt(root: THREE.Object3D, point: THREE.Vector3, parts: THREE.Object3D[]): THREE.Object3D {
  const pivot = new THREE.Object3D();
  pivot.position.copy(point);
  root.add(pivot);
  parts.forEach((p) => pivot.attach(p));
  return pivot;
}

interface Rig {
  root: THREE.Group;
  arm: THREE.Object3D;
  needle: THREE.Object3D;
}

async function buildRig(): Promise<Rig> {
  const root = (await loadModel()).clone(true);
  const arm = pivotAt(root, SHOULDER, findParts(root, WAVE_PARTS));
  const needle = pivotAt(root, COMPASS_CENTRE, findParts(root, NEEDLE_PARTS));
  return { root, arm, needle };
}

function buildScene(rig: Rig, framing: MascotFraming, aspect = 1) {
  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x334155, 2.2));
  const key = new THREE.DirectionalLight(0xffffff, 2.4);
  key.position.set(2, 4, 5);
  scene.add(key);
  scene.add(rig.root);
  // whole body is about 3.1 wide x 2.9 tall around (0.07, 1.43); the portrait keeps roof, face and compass
  const target = framing === 'full' ? new THREE.Vector3(0.07, 1.43, 0) : new THREE.Vector3(0.0, 1.6, 0);
  const height = framing === 'full' ? 3.4 : 2.5;
  const camera = new THREE.PerspectiveCamera(30, aspect, 0.1, 50);
  const distance = height / 2 / Math.tan(THREE.MathUtils.degToRad(15)) / Math.min(1, aspect);
  camera.position.set(target.x + distance * 0.18, target.y + distance * 0.08, distance);
  camera.lookAt(target);
  return { scene, camera };
}

function animate(rig: Rig, mood: MascotMood, t: number) {
  rig.root.position.y = Math.sin(t * 2.2) * 0.04;
  rig.root.rotation.y = Math.sin(t * 0.7) * (mood === 'thinking' ? 0.12 : 0.22);
  // a burst of waving every 5 s, otherwise a slight sway
  const waving = mood === 'wave' && t % 5 < 2.2;
  rig.arm.rotation.z = waving ? 0.25 + Math.sin(t * 9) * 0.35 : Math.sin(t * 1.5) * 0.06;
  rig.needle.rotation.z = mood === 'thinking' ? -t * 3.2 : Math.sin(t * 1.1) * 0.35;
}

const reducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

/** Live, animated mascot. Keep to one or two on screen at a time. */
export function Mascot3D({
  size = 120,
  mood = 'wave',
  framing = 'full',
  className,
}: {
  size?: number;
  mood?: MascotMood;
  framing?: MascotFraming;
  className?: string;
}) {
  const hostRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    let disposed = false;
    let frame = 0;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(size, size);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(renderer.domElement);

    buildRig()
      .then((rig) => {
        if (disposed) return;
        const { scene, camera } = buildScene(rig, framing);
        const clock = new THREE.Clock();
        const still = reducedMotion();
        const tick = () => {
          animate(rig, mood, still ? 0.9 : clock.getElapsedTime());
          renderer.render(scene, camera);
          if (!still) frame = requestAnimationFrame(tick);
        };
        tick();
      })
      .catch((err) => console.warn('[Mascot] model failed to load', err));

    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
    };
  }, [size, mood, framing]);

  return (
    <div
      ref={hostRef}
      className={className}
      style={{ width: size, height: size, flexShrink: 0 }}
      role="img"
      aria-label="HDB-Compass mascot"
    />
  );
}

// One still image per framing, rendered once off-screen and reused for every avatar and the favicon
const snapshots = new Map<MascotFraming, Promise<string>>();
function snapshot(framing: MascotFraming): Promise<string> {
  let p = snapshots.get(framing);
  if (!p) {
    p = buildRig().then((rig) => {
      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
      renderer.setSize(256, 256);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      const { scene, camera } = buildScene(rig, framing);
      animate(rig, 'idle', 0.9);
      renderer.render(scene, camera);
      const url = renderer.domElement.toDataURL('image/png');
      renderer.dispose();
      renderer.forceContextLoss();
      return url;
    });
    snapshots.set(framing, p);
  }
  return p;
}

export function useMascotSnapshot(framing: MascotFraming = 'portrait'): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    snapshot(framing).then((u) => live && setUrl(u)).catch(() => undefined);
    return () => {
      live = false;
    };
  }, [framing]);
  return url;
}

/** Still mascot for the logo and chat avatars; shows `fallback` until the image is ready. */
export function MascotAvatar({
  size = 26,
  framing = 'portrait',
  fallback = null,
}: {
  size?: number;
  framing?: MascotFraming;
  fallback?: React.ReactNode;
}) {
  const url = useMascotSnapshot(framing);
  if (!url) return <>{fallback}</>;
  return <img src={url} width={size} height={size} alt="HDB-Compass mascot" style={{ display: 'block' }} />;
}

/** Use the mascot as the browser tab icon once its image is ready. */
export function useMascotFavicon() {
  const url = useMascotSnapshot('portrait');
  useEffect(() => {
    if (!url) return;
    let link = document.querySelector<HTMLLinkElement>('link[rel="icon"]');
    if (!link) {
      link = document.createElement('link');
      link.rel = 'icon';
      document.head.appendChild(link);
    }
    link.type = 'image/png';
    link.href = url;
  }, [url]);
}
