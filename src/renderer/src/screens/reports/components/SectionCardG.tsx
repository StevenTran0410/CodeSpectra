import React, { useState } from 'react'
import type { SectionG, SectionRadarSlot } from '../../../types/analysis'
import SectionCard, { type SectionCardRerunProps } from './SectionCard'

function lastTwoSegments(path: string): string {
  const norm = path.replace(/\\/g, '/').replace(/^\/+/, '')
  const parts = norm.split('/').filter(Boolean)
  if (parts.length <= 2)
    return norm
  return parts.slice(-2).join('/')
}

const SLOTS: { key: keyof SectionG; label: string }[] = [
  { key: 'entrypoint', label: 'Entrypoint' },
  { key: 'backbone', label: 'Backbone' },
  { key: 'critical_config', label: 'Critical Config' },
  { key: 'highest_centrality', label: 'Highest Centrality' },
  { key: 'most_dangerous_to_touch', label: 'Most Dangerous' },
  { key: 'read_first', label: 'Read First' },
]

export default function SectionCardG({
  data,
  onRerun,
  rerunBusy,
}: { data: SectionG } & SectionCardRerunProps): React.ReactElement {
  const [showOther, setShowOther] = useState(false)
  const conf = data.confidence === 'high' || data.confidence === 'medium' || data.confidence === 'low'
    ? data.confidence
    : 'medium'

  const renderSlot = (label: string, slot: SectionRadarSlot | undefined) => {
    const s = slot ?? { file: '', reason: '' }
    const path = lastTwoSegments(s.file || '')
    return (
      <div className="border border-zinc-800 rounded p-2 bg-zinc-900/30" title={s.file}>
        <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-1">{label}</div>
        <div className="font-mono text-xs text-indigo-300 truncate">{path || '—'}</div>
        {s.reason ? (
          <div className="text-[11px] text-zinc-400 mt-1 leading-snug line-clamp-3" title={s.reason}>
            {s.reason}
          </div>
        ) : null}
      </div>
    )
  }

  return (
    <SectionCard
      sectionId="G"
      sectionName="Important Files Radar"
      confidence={conf}
      evidenceCount={data.evidence_files?.length ?? 0}
      onRerun={onRerun}
      rerunBusy={rerunBusy}
    >
      <div className="grid grid-cols-2 gap-2">
        {SLOTS.map(({ key, label }) => (
          <div key={key}>{renderSlot(label, data[key] as SectionRadarSlot)}</div>
        ))}
      </div>
      {data.other_important && data.other_important.length > 0 && (
        <div className="mt-3 border-t border-zinc-800 pt-2">
          <button
            type="button"
            onClick={() => setShowOther((v) => !v)}
            className="text-[11px] text-zinc-400 hover:text-zinc-200"
          >
            {showOther ? '▼' : '▸'} Other important ({data.other_important.length})
          </button>
          {showOther && (
            <ul className="mt-2 space-y-2">
              {data.other_important.map((o, i) => (
                <li key={i} className="text-xs border border-zinc-800 rounded p-2" title={o.file}>
                  <div className="font-mono text-indigo-300 truncate">{lastTwoSegments(o.file)}</div>
                  {o.reason ? <div className="text-zinc-400 mt-0.5">{o.reason}</div> : null}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </SectionCard>
  )
}
