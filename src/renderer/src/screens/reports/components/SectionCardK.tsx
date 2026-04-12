import React from 'react'
import type { SectionK } from '../../../types/analysis'
import { normConf } from '../../../lib/reportUtils'
import SectionCard, { type SectionCardRerunProps } from './SectionCard'

const SECTION_LETTERS = 'ABCDEFGHIJ'.split('') as string[]

function scoreChipClass(level: string | undefined): string {
  if (level === 'high') return 'bg-emerald-900/50 text-emerald-300 border-emerald-800'
  if (level === 'medium') return 'bg-yellow-900/50 text-yellow-300 border-yellow-800'
  if (level === 'low') return 'bg-rose-900/50 text-rose-300 border-rose-800'
  return 'bg-zinc-800/50 text-zinc-400 border-zinc-700'
}

export default function SectionCardK({
  data,
  onExportAudit,
  exportAuditBusy,
  onRerun,
  rerunBusy,
}: {
  data: SectionK
  onExportAudit?: () => void | Promise<void>
  exportAuditBusy?: boolean
} & SectionCardRerunProps): React.ReactElement {
  const conf = normConf(data.overall_confidence)
  const pct = Number.isFinite(data.coverage_percentage) ? data.coverage_percentage : 0
  const clamped = Math.max(0, Math.min(100, pct))

  return (
    <SectionCard
      sectionId="K"
      sectionName="Confidence Auditor"
      confidence={conf}
      defaultCollapsed
      onRerun={onRerun}
      rerunBusy={rerunBusy}
      headerExtra={
        onExportAudit ? (
          <button
            type="button"
            disabled={exportAuditBusy}
            onClick={() => {
              void onExportAudit()
            }}
            className="px-2 py-0.5 text-[10px] rounded border border-indigo-700 text-indigo-300 hover:border-indigo-500 disabled:opacity-40"
          >
            {exportAuditBusy ? '…' : 'Export Audit'}
          </button>
        ) : undefined
      }
    >
      <div className="space-y-3">
        <div>
          <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-1.5">
            Coverage (high + medium sections)
          </div>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-2 rounded-full bg-zinc-800 overflow-hidden">
              <div
                className="h-full bg-emerald-600/90 rounded-full transition-all"
                style={{ width: `${clamped}%` }}
              />
            </div>
            <span className="text-xs font-mono text-zinc-300 tabular-nums">
              {clamped.toFixed(0)}%
            </span>
          </div>
        </div>

        <div>
          <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-1.5">Section scores</div>
          <div className="flex flex-wrap gap-1">
            {SECTION_LETTERS.map((letter) => {
              const raw = data.section_scores?.[letter as keyof typeof data.section_scores]
              const level =
                typeof raw === 'string' ? raw.toLowerCase() : 'unknown'
              return (
                <span
                  key={letter}
                  className={`text-[10px] px-1.5 py-0.5 rounded border font-mono font-semibold ${scoreChipClass(
                    level === 'high' || level === 'medium' || level === 'low' ? level : undefined
                  )}`}
                >
                  {letter}
                </span>
              )
            })}
          </div>
        </div>

        {data.weakest_sections && data.weakest_sections.length > 0 && (
          <div className="rounded border border-rose-900/40 bg-rose-950/25 px-2 py-2">
            <div className="text-[10px] uppercase tracking-wide text-rose-300/90 mb-1.5">
              Weakest sections
            </div>
            <div className="flex flex-wrap gap-1">
              {data.weakest_sections.map((s) => (
                <span
                  key={s}
                  className="text-[10px] px-1.5 py-0.5 rounded border border-rose-800/60 bg-rose-900/30 text-rose-200 font-mono"
                >
                  {s}
                </span>
              ))}
            </div>
          </div>
        )}

        {data.notes ? (
          <p className="text-sm text-zinc-300 whitespace-pre-wrap">{data.notes}</p>
        ) : null}

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
