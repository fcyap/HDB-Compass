import { useCallback, useState } from 'react';

export interface ShortlistItem {
  block: string;      // clean display name, e.g. "Blk 231 TAMPINES ST 21"
  flat_type: string;  // e.g. "4 ROOM"
}

const KEY = 'hdb-compass-shortlist';

export const blockLabel = (name: string): string => {
  const s = name.replace(/\s*\(\d+(?:\.\d+)?m\)/, '').trim();
  return /^(blk|block)\s/i.test(s) ? s : `Blk ${s}`;
};

const sameItem = (a: ShortlistItem, b: ShortlistItem) =>
  a.block.toUpperCase() === b.block.toUpperCase() && a.flat_type.toUpperCase() === b.flat_type.toUpperCase();

const load = (): ShortlistItem[] => {
  try {
    const raw = localStorage.getItem(KEY);
    const data = raw ? JSON.parse(raw) : [];
    return Array.isArray(data)
      ? data.filter((d) => d && typeof d.block === 'string' && typeof d.flat_type === 'string')
      : [];
  } catch {
    return [];
  }
};

const save = (items: ShortlistItem[]) => {
  try {
    localStorage.setItem(KEY, JSON.stringify(items));
  } catch {
    /* storage unavailable: the shortlist lasts for this visit only */
  }
};

export function joinNames(names: string[]): string {
  if (names.length <= 1) return names.join('');
  return `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
}

export function compareText(items: ShortlistItem[]): string {
  return `Compare ${joinNames(items.map((i) => i.block))} for ${items[0].flat_type}`;
}

export function useShortlist() {
  const [items, setItems] = useState<ShortlistItem[]>(load);
  const update = useCallback((fn: (prev: ShortlistItem[]) => ShortlistItem[]) => {
    setItems((prev) => {
      const next = fn(prev);
      save(next);
      return next;
    });
  }, []);
  const has = useCallback((it: ShortlistItem) => items.some((x) => sameItem(x, it)), [items]);
  const toggle = useCallback(
    (it: ShortlistItem) =>
      update((prev) => (prev.some((x) => sameItem(x, it)) ? prev.filter((x) => !sameItem(x, it)) : [...prev, it])),
    [update]
  );
  const remove = useCallback((it: ShortlistItem) => update((prev) => prev.filter((x) => !sameItem(x, it))), [update]);
  const clear = useCallback(() => update(() => []), [update]);
  return { items, has, toggle, remove, clear };
}
