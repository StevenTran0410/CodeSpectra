import React from 'react'
import type { SectionB } from '../../../types/analysis'
import SectionCard, { type SectionCardRerunProps } from './SectionCard'

function normConf(c: string | undefined): 'high' | 'medium' | 'low' {
  return c === 'high' || c === 'medium' || c === 'low' ? c : 'medium'
}

export default function SectionCardB({
  data,
  onRerun,
  rerunBusy,
}: { data: SectionB } & SectionCardRerunProps): React.ReactElement {
  const conf = normConf(data.confidence)
  return (
    <SectionCard
      sectionId="B"
      sectionName="Architecture Overview"
      confidence={conf}
      evidenceCount={data.evidence_files?.length ?? 0}
      onRerun={onRerun}
      rerunBusy={rerunBusy}
    >
      <div className="space-y-3">
        {data.main_layers && data.main_layers.length > 0 && (
          <div>
            <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-1.5">Layers</div>
            <div className="flex flex-wrap gap-1">
              {data.main_layers.map((l) => (
                <span
                  key={l}
                  className="text-[10px] px-1.5 py-0.5 rounded border border-zinc-700 bg-zinc-900/40 text-zinc-300 font-mono"
                >
                  {l}
                </span>
              ))}
            </div>
          </div>
        )}
        {data.frameworks && data.frameworks.length > 0 && (
          <div>
            <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-1.5">
              Frameworks
            </div>
            <div className="flex flex-wrap gap-1">
              {data.frameworks.map((f) => (
                <span
                  key={f}
                  className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-900/50 text-indigo-200 border border-indigo-800/60"
                >
                  {f}
                </span>
              ))}
            </div>
          </div>
        )}
        {data.entrypoints && data.entrypoints.length > 0 && (
          <div>
            <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-1.5">
              Entrypoints
            </div>
            <ul className="space-y-0.5">
              {data.entrypoints.map((p) => (
                <li key={p} className="font-mono text-xs text-zinc-300 pl-2 -indent-2">
                  <span className="text-zinc-600 mr-1">—</span>
                  {p}
                </li>
              ))}
            </ul>
          </div>
        )}
        {data.main_services && data.main_services.length > 0 && (
          <div>
            <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-1.5">
              Main services
            </div>
            <div className="rounded border border-zinc-800 overflow-hidden text-xs">
              <div className="grid grid-cols-[1fr_1.2fr_1fr] gap-2 px-2 py-1.5 bg-zinc-900/50 text-[10px] uppercase tracking-wide text-zinc-500 border-b border-zinc-800">
                <div>Name</div>
                <div>Path</div>
                <div>Role</div>
              </div>
              {data.main_services.map((s, i) => (
                <div
                  key={`${s.path}-${i}`}
                  className={`grid grid-cols-[1fr_1.2fr_1fr] gap-2 px-2 py-1.5 border-b border-zinc-800/60 last:border-b-0 ${
                    i % 2 === 1 ? 'bg-zinc-900/30' : ''
                  }`}
                >
                  <div className="text-zinc-200 font-semibold truncate" title={s.name}>
                    {s.name || '—'}
                  </div>
                  <div className="font-mono text-[11px] text-zinc-400 truncate" title={s.path}>
                    {s.path || '—'}
                  </div>
                  <div className="text-zinc-500 text-[11px] truncate" title={s.role}>
                    {s.role || '—'}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        {data.external_integrations && data.external_integrations.length > 0 && (
          <div>
            <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-1.5">
              External integrations
            </div>
            <div className="flex flex-wrap gap-1">
              {data.external_integrations.map((x) => (
                <span
                  key={x}
                  className="text-[10px] px-1.5 py-0.5 rounded border border-zinc-700 bg-zinc-800/60 text-zinc-400"
                >
                  {x}
                </span>
              ))}
            </div>
          </div>
        )}
        {data.database_hints && data.database_hints.length > 0 && (
          <div>
            <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-1.5">
              Database hints
            </div>
            <ul className="space-y-0.5">
              {data.database_hints.map((h) => (
                <li key={h} className="italic text-xs text-zinc-500">
                  {h}
                </li>
              ))}
            </ul>
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
