import React from 'react'
import type { ConventionAspect, SectionD } from '../../../types/analysis'
import SectionCard, { type SectionCardRerunProps } from './SectionCard'

function normConf(c: string | undefined): 'high' | 'medium' | 'low' {
  return c === 'high' || c === 'medium' || c === 'low' ? c : 'medium'
}

const ROWS: { key: keyof Pick<
  SectionD,
  | 'naming_style'
  | 'error_handling'
  | 'async_style'
  | 'di_style'
  | 'class_vs_functional'
  | 'test_style'
>; label: string }[] = [
  { key: 'naming_style', label: 'Naming style' },
  { key: 'error_handling', label: 'Error handling' },
  { key: 'async_style', label: 'Async style' },
  { key: 'di_style', label: 'Dependency injection' },
  { key: 'class_vs_functional', label: 'Class vs functional' },
  { key: 'test_style', label: 'Test style' },
]

function aspectParts(value: ConventionAspect | string): { desc: string; files: string[] } {
  if (typeof value === 'string') {
    return { desc: value, files: [] }
  }
  return {
    desc: value?.description ?? '',
    files: Array.isArray(value?.evidence_files) ? value.evidence_files : [],
  }
}

export default function SectionCardD({
  data,
  onRerun,
  rerunBusy,
}: { data: SectionD } & SectionCardRerunProps): React.ReactElement {
  const conf = normConf(data.confidence)
  return (
    <SectionCard
      sectionId="D"
      sectionName="Coding Conventions"
      confidence={conf}
      evidenceCount={data.evidence_files?.length ?? 0}
      onRerun={onRerun}
      rerunBusy={rerunBusy}
    >
      <div className="space-y-3">
        <div className="rounded border border-zinc-800 overflow-hidden text-xs">
          <div className="grid grid-cols-[minmax(0,0.85fr)_1fr_minmax(0,1fr)] gap-2 px-2 py-1.5 bg-zinc-900/50 text-[10px] uppercase tracking-wide text-zinc-500 border-b border-zinc-800">
            <div>Category</div>
            <div>Description</div>
            <div>Evidence files</div>
          </div>
          {ROWS.map(({ key, label }, i) => {
            const { desc, files } = aspectParts(data[key] as ConventionAspect | string)
            return (
              <div
                key={key}
                className={`grid grid-cols-[minmax(0,0.85fr)_1fr_minmax(0,1fr)] gap-2 px-2 py-1.5 border-b border-zinc-800/60 last:border-b-0 ${
                  i % 2 === 1 ? 'bg-zinc-900/30' : ''
                }`}
              >
                <div className="text-[10px] uppercase tracking-wider text-zinc-500">{label}</div>
                <div className="text-zinc-200 text-xs">{desc.trim() ? desc : '—'}</div>
                <div className="font-mono text-[10px] text-zinc-500 break-all">
                  {files.length > 0 ? files.join(', ') : '—'}
                </div>
              </div>
            )
          })}
        </div>

        <div>
          <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-1.5">
            Convention signals ({data.signals?.length ?? 0})
          </div>
          {!data.signals || data.signals.length === 0 ? (
            <div className="text-zinc-500 italic text-xs">No signals detected</div>
          ) : (
            <div className="rounded border border-zinc-800 overflow-hidden">
              {data.signals.map((s, idx) => (
                <div
                  key={`${s.category}-${idx}`}
                  className={`grid grid-cols-[auto_1fr_minmax(0,1fr)] gap-2 px-2 py-1.5 border-b border-zinc-800/60 last:border-b-0 items-start ${
                    idx % 2 === 1 ? 'bg-zinc-900/20' : ''
                  }`}
                >
                  <span className="text-[10px] px-1.5 py-0.5 rounded border border-zinc-700 bg-zinc-800 text-zinc-300 shrink-0">
                    {s.category || '—'}
                  </span>
                  <span className="text-xs text-zinc-200">{s.description || '—'}</span>
                  <span className="font-mono text-[10px] text-zinc-500 break-all">
                    {s.evidence || '—'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

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
