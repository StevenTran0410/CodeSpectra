import React, { useState } from 'react'
import type { ForbiddenRule, SectionE } from '../../../types/analysis'
import SectionCard from './SectionCard'

function normConf(c: string | undefined): 'high' | 'medium' | 'low' {
  return c === 'high' || c === 'medium' || c === 'low' ? c : 'medium'
}

function ruleSeverityClass(sev: string | undefined): string {
  const s = (sev || 'weak').toLowerCase()
  if (s === 'strong')
    return 'bg-emerald-900/50 text-emerald-300 border-emerald-800'
  if (s === 'suspected')
    return 'bg-amber-900/50 text-amber-300 border-amber-800'
  return 'bg-zinc-800 text-zinc-400 border-zinc-700'
}

function violSeverityClass(sev: string | undefined): string {
  const s = (sev || 'medium').toLowerCase()
  if (s === 'high') return 'bg-rose-900/50 text-rose-200 border-rose-800'
  if (s === 'low') return 'bg-zinc-800 text-zinc-400 border-zinc-700'
  return 'bg-amber-900/40 text-amber-200 border-amber-800'
}

export default function SectionCardE({ data }: { data: SectionE }): React.ReactElement {
  const conf = normConf(data.confidence)
  const [violOpen, setViolOpen] = useState(false)
  const nViol = data.violations_found?.length ?? 0

  return (
    <SectionCard
      sectionId="E"
      sectionName="Forbidden Things"
      confidence={conf}
      evidenceCount={data.evidence_files?.length ?? 0}
    >
      <div className="space-y-3">
        <div>
          <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-1.5">
            Inferred rules ({data.rules?.length ?? 0})
          </div>
          {!data.rules || data.rules.length === 0 ? (
            <div className="text-zinc-500 italic text-xs">No rules inferred</div>
          ) : (
            <ul className="space-y-2">
              {data.rules.map((r: ForbiddenRule, i: number) => (
                <li key={i} className="rounded border border-zinc-800/80 px-2 py-1.5">
                  <div className="flex flex-wrap items-center gap-2">
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded border ${ruleSeverityClass(
                        r.severity
                      )}`}
                    >
                      {(r.severity || 'weak').toLowerCase()}
                    </span>
                    <span className="text-xs text-zinc-200 font-medium">{r.rule || '—'}</span>
                  </div>
                  {r.inferred_from ? (
                    <div className="text-[11px] text-zinc-500 italic mt-1 pl-0.5">
                      {r.inferred_from}
                    </div>
                  ) : null}
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="rounded border border-zinc-800 overflow-hidden">
          <button
            type="button"
            className="w-full flex items-center justify-between gap-2 px-2 py-1.5 bg-zinc-900/40 text-left text-[10px] uppercase tracking-wide text-zinc-400 hover:bg-zinc-900/60"
            onClick={() => setViolOpen((o) => !o)}
            aria-expanded={violOpen}
          >
            <span>Violations found ({nViol})</span>
            <span className="text-zinc-500">{violOpen ? '▾' : '▸'}</span>
          </button>
          {violOpen && (
            <div className="px-2 py-2 space-y-2 border-t border-zinc-800">
              {!data.violations_found || data.violations_found.length === 0 ? (
                <div className="text-zinc-500 italic text-xs">No violations listed</div>
              ) : (
                data.violations_found.map((v, i) => (
                  <div
                    key={i}
                    className="rounded border border-zinc-800/60 px-2 py-1.5 space-y-1"
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-[10px] bg-red-950 text-red-300 px-1 rounded">
                        {v.rule || '—'}
                      </span>
                      <span
                        className={`text-[10px] px-1.5 py-0.5 rounded border ${violSeverityClass(
                          v.severity
                        )}`}
                      >
                        {(v.severity || 'medium').toLowerCase()}
                      </span>
                    </div>
                    <div className="font-mono text-[10px] text-zinc-400 break-all">
                      {v.file || '—'}
                    </div>
                    {v.description ? (
                      <div className="text-xs text-zinc-300">{v.description}</div>
                    ) : null}
                  </div>
                ))
              )}
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
