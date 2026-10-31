import React from 'react';
import { ShortlistItem, compareText } from '../shortlist';

interface Props {
  items: ShortlistItem[];
  disabled: boolean;
  onRemove: (it: ShortlistItem) => void;
  onClear: () => void;
  onCompare: (text: string) => void;
}

export const ShortlistTray: React.FC<Props> = ({ items, disabled, onRemove, onClear, onCompare }) => {
  if (items.length === 0) return null;
  const types = Array.from(new Set(items.map((i) => i.flat_type.toUpperCase())));
  const mixed = types.length > 1;
  // compare the first four of one flat type; the tray says so when more are saved
  const group = mixed ? [] : items.slice(0, 4);
  const canCompare = group.length >= 2;
  const note = items.length < 2
    ? 'Add one more block to compare.'
    : mixed
      ? 'Compare needs blocks of the same flat type.'
      : items.length > 4
        ? 'Comparison covers 4 blocks at a time; the first 4 will be used.'
        : '';

  return (
    <div className="shortlist-tray" role="region" aria-label="Shortlist">
      <div className="shortlist-head">
        <span className="shortlist-title">Shortlist ({items.length})</span>
        <span className="shortlist-actions">
          <button type="button" className="shortlist-btn" onClick={onClear}>Clear</button>
          <button
            type="button"
            className="shortlist-btn shortlist-compare"
            disabled={disabled || !canCompare}
            onClick={() => onCompare(compareText(group))}
          >
            Compare ({canCompare ? group.length : items.length})
          </button>
        </span>
      </div>
      <ul className="shortlist-list">
        {items.map((it) => (
          <li key={`${it.block}|${it.flat_type}`} className="shortlist-item">
            <span className="shortlist-name" title={`${it.block} (${it.flat_type})`}>{it.block}</span>
            <span className="shortlist-type">{it.flat_type}</span>
            <button type="button" className="shortlist-remove" aria-label={`Remove ${it.block}`} onClick={() => onRemove(it)}>×</button>
          </li>
        ))}
      </ul>
      {note && <div className="shortlist-note">{note}</div>}
    </div>
  );
};
