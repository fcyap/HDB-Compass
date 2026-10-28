import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

/**
 * HDB-Compass mascot: a little house holding a compass (public/mascot/hdb-compass-home-guide.glb).
 *
 * The model's parts are flat at the origin with their positions baked in, so the wave, the compass needle and the eyes
 * are animated by re-parenting those parts onto pivots (shoulder, compass centre, each eye's centre). The body stays
 * still; the eyes blink and glance around at random, and the arm waves now and then.
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
const EYES = [
  { centre: new THREE.Vector3(-0.39, 1.68, 0.32), parts: ['Left eye', 'Left eye glint'], brow: 'Left eyebrow' },
  { centre: new THREE.Vector3(0.39, 1.68, 0.32), parts: ['Right eye', 'Right eye glint'], brow: 'Right eyebrow' },
];
const GAZE_RANGE = { x: 0.05, y: 0.03 };   // how far the eyes slide on the face when looking around

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
  eyes: { pivot: THREE.Object3D; home: THREE.Vector3 }[];
  brows: { part: THREE.Object3D; home: number }[];
}

async function buildRig(): Promise<Rig> {
  const root = (await loadModel()).clone(true);
  const arm = pivotAt(root, SHOULDER, findParts(root, WAVE_PARTS));
  const needle = pivotAt(root, COMPASS_CENTRE, findParts(root, NEEDLE_PARTS));
  const eyes = EYES.map((e) => {
    const pivot = pivotAt(root, e.centre, findParts(root, e.parts));
    return { pivot, home: pivot.position.clone() };
  });
  const brows = EYES.flatMap((e) => findParts(root, [e.brow])).map((part) => ({ part, home: part.position.y }));
  return { root, arm, needle, eyes, brows };
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

const smoothstep = (a: number, b: number, x: number) => {
  const k = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return k * k * (3 - 2 * k);
};

const WAVE_CYCLE = 6;   // seconds between the starts of two greetings
const WAVE_LENGTH = 2.6; // seconds the arm is up, including raising and lowering it

function animate(rig: Rig, mood: MascotMood, t: number) {
  // The body stays put. The arm eases up, waves and eases down again: every pose is a continuous function of time,
  // so no frame jumps at the start or end of a greeting.
  const p = (t + WAVE_CYCLE - 1.2) % WAVE_CYCLE;   // first greeting a moment after the mascot appears
  const up = mood === 'wave' ? smoothstep(0, 0.5, p) * (1 - smoothstep(WAVE_LENGTH - 0.6, WAVE_LENGTH, p)) : 0;
  const wave = Math.sin(Math.min(p, WAVE_LENGTH) * 2 * Math.PI * 1.6) * 0.28;
  rig.arm.rotation.z = up * (0.35 + wave);
  rig.needle.rotation.z = mood === 'thinking' ? -t * 3.2 : Math.sin(t * 0.9) * 0.25;
}

/**
 * Blinks and glances, driven by randomness like a person's: a blink every 2-6 s (sometimes a double blink), and a
 * quick eye movement to a new spot every 1.2-4 s, returning to the centre about half the time. A large glance often
 * comes with a blink. While "thinking" the eyes favour the compass and a ponder up to the side.
 */
class Eyes {
  private gaze = new THREE.Vector2();
  private target = new THREE.Vector2();
  private nextGlance = 0.8;
  private nextBlink = 1.6 + Math.random() * 2;
  private blinkStart = -1;
  private queuedBlink = -1;

  constructor(private mood: MascotMood) {}

  update(rig: Rig, t: number, dt: number) {
    if (t >= this.nextGlance) this.glance(t);
    if (t >= this.nextBlink) this.blink(t, Math.random() < 0.18);
    if (this.queuedBlink > 0 && t >= this.queuedBlink) {
      this.blinkStart = t;
      this.queuedBlink = -1;
    }
    // saccade: fast exponential approach (about 60 ms), then hold still
    this.gaze.lerp(this.target, 1 - Math.exp(-dt / 0.06));
    const lid = this.lidClosure(t);
    rig.eyes.forEach(({ pivot, home }) => {
      pivot.position.set(home.x + this.gaze.x * GAZE_RANGE.x, home.y + this.gaze.y * GAZE_RANGE.y, home.z);
      pivot.scale.y = 1 - 0.9 * lid;
    });
    // brows follow the eyes a little when looking up, and dip with a blink
    rig.brows.forEach(({ part, home }) => (part.position.y = home + this.gaze.y * 0.015 - lid * 0.012));
  }

  private glance(t: number) {
    const r = Math.random();
    if (this.mood === 'thinking') {
      this.target.set(...((r < 0.55 ? [-0.9, -0.7] : r < 0.85 ? [0.6, 0.8] : [0, 0]) as [number, number]));
    } else if (r < 0.5) {
      this.target.set(0, 0);
    } else {
      const a = Math.random() * Math.PI * 2;
      const d = 0.5 + Math.random() * 0.5;
      this.target.set(Math.cos(a) * d, Math.sin(a) * d * 0.7);
    }
    if (this.gaze.distanceTo(this.target) > 0.8 && Math.random() < 0.35) this.blink(t, false);
    this.nextGlance = t + 1.2 + Math.random() * 2.8;
  }

  private blink(t: number, double: boolean) {
    this.blinkStart = t;
    this.queuedBlink = double ? t + 0.28 : -1;
    this.nextBlink = t + 2 + Math.random() * 4;
  }

  // 0 = open, 1 = shut: close in 70 ms, stay shut 40 ms, open in 100 ms
  private lidClosure(t: number) {
    if (this.blinkStart < 0) return 0;
    const e = t - this.blinkStart;
    if (e < 0.07) return smoothstep(0, 0.07, e);
    if (e < 0.11) return 1;
    if (e < 0.21) return 1 - smoothstep(0.11, 0.21, e);
    return 0;
  }
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
        const eyes = new Eyes(mood);
        let last = 0;
        const tick = () => {
          const t = still ? 0 : clock.getElapsedTime();
          animate(rig, still ? 'idle' : mood, t);
          if (!still) eyes.update(rig, t, Math.min(0.1, t - last));
          last = t;
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
      animate(rig, 'idle', 0);
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
