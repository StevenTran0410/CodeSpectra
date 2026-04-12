import React from 'react'
import type { FolderRole, SectionC } from '../../../types/analysis'
import { normConf } from '../../../lib/reportUtils'
import SectionCard, { type SectionCardRerunProps } from './SectionCard'

const ROLE_BADGE: Record<string, string> = {
  domain: 'bg-indigo-900/50 text-indigo-300 border-indigo-800/60',
  infrastructure: 'bg-zinc-800 text-zinc-300 border-zinc-700',
  delivery: 'bg-emerald-900/50 text-emerald-300 border-emerald-800/60',
  shared: 'bg-blue-900/50 text-blue-300 border-blue-800/60',
  test: 'bg-amber-900/40 text-amber-300 border-amber-800/50',
  generated: 'bg-purple-900/50 text-purple-300 border-purple-800/60',
  unknown: 'bg-zinc-800/80 text-zinc-500 border-zinc-700',
}

function roleClass(role: FolderRole | string | undefined): string {
  const r = String(role ?? 'unknown')
  return ROLE_BADGE[r] ?? ROLE_BADGE.unknown
}

export default function SectionCardC({
  data,
  onRerun,
  rerunBusy,
}: { data: SectionC } & SectionCardRerunProps): React.ReactElement {
  const conf = normConf(data.confidence)
  return (
    <SectionCard
      sectionId="C"
      sectionName="Repo Structure"
      confidence={conf}
      evidenceCount={data.evidence_files?.length ?? 0}
      onRerun={onRerun}
      rerunBusy={rerunBusy}
    >
      <div className="space-y-3">
        {data.summary ? (
          <p className="text-sm text-zinc-300 leading-relaxed">{data.summary}</p>
        ) : null}
        {data.folders && data.folders.length > 0 && (
          <div className="rounded border border-zinc-800 overflow-hidden">
            <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1.2fr)] gap-2 px-2 py-1.5 bg-zinc-900/50 text-[10px] uppercase tracking-wide text-zinc-500 border-b border-zinc-800">
              <div>Path</div>
              <div>Role</div>
              <div>Description</div>
            </div>
            {data.folders.map((f, i) => (
              <div
                key={`${f.path}-${i}`}
                className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1.2fr)] gap-2 px-2 py-2 border-b border-zinc-800/60 last:border-b-0 items-start"
              >
                <div className="font-mono text-xs text-zinc-300 break-all" title={f.path}>
                  {f.path}
                </div>
                <div>
                  <span
                    className={`inline-block text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded border ${roleClass(f.role)}`}
                  >
                    {String(f.role ?? 'unknown')}
                  </span>
                </div>
                <div className="text-xs text-zinc-400 leading-snug">{f.description || '—'}</div>
              </div>
            ))}
          </div>
        )}
        {data.blind_spots && data.blind_spots.length > 0 && (
          <div className="rounded border border-amber-900/40 bg-amber-950/20 px-2 py-2 text-xs text-amber-200/90">
            <div className="font-semibold text-amber-400/90 mb-1">Blind spots</div>
            <ul className="list-disc pl-4 space-y-0.5 text-amber-100/80">
              {data.blind_spots.map((b, i) => (
                <li key={i}>{b}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </SectionCard>
  )
}
