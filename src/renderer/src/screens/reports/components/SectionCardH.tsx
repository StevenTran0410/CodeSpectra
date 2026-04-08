import React from 'react'
import type { SectionH } from '../../../types/analysis'
import SectionCard, { type SectionCardRerunProps } from './SectionCard'

function normConf(c: string | undefined): 'high' | 'medium' | 'low' {
  return c === 'high' || c === 'medium' || c === 'low' ? c : 'medium'
}

export default function SectionCardH({
  data,
  onRerun,
  rerunBusy,
}: { data: SectionH } & SectionCardRerunProps): React.ReactElement {
  const conf = normConf(data.confidence)
  const ordered = [...(data.steps ?? [])].sort((a, b) => (a.order ?? 0) - (b.order ?? 0))

  return (
    <SectionCard
      sectionId="H"
      sectionName="Onboarding Reading Order"
      confidence={conf}
      evidenceCount={data.evidence_files?.length ?? 0}
      onRerun={onRerun}
      rerunBusy={rerunBusy}
    >
      <div className="space-y-3">
        <div className="text-[11px] text-zinc-500 mb-2">
          <span className="inline-flex px-2 py-0.5 rounded bg-zinc-800/80 border border-zinc-700 text-zinc-400">
            Est. {data.total_estimated_minutes ?? 30} min total
          </span>
        </div>
        {ordered.length > 0 && (
          <ol className="space-y-3 list-none p-0 m-0">
            {ordered.map((s, idx) => {
              const n = s.order && s.order > 0 ? s.order : idx + 1
              return (
                <li key={`${s.file}-${n}-${idx}`} className="flex gap-2 items-start">
                  <span className="inline-flex w-6 h-6 rounded-full bg-indigo-900/60 text-indigo-200 border border-indigo-800/70 text-[11px] font-semibold items-center justify-center shrink-0">
                    {n}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="font-mono text-xs text-zinc-300 break-all" title={s.file}>
                      {s.file || '—'}
                    </div>
                    {s.goal ? (
                      <div className="text-xs font-semibold text-zinc-200 mt-0.5">{s.goal}</div>
                    ) : null}
                    {s.outcome ? (
                      <div className="text-xs text-zinc-400 mt-0.5">{s.outcome}</div>
                    ) : null}
                  </div>
                </li>
              )
            })}
          </ol>
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
