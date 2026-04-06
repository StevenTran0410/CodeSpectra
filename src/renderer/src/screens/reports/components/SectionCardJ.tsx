import React from 'react'
import type { RiskFinding, SectionJ } from '../../../types/analysis'
import { SEVERITY_COLORS, SEVERITY_DOT } from '../constants'
import SectionCard from './SectionCard'

export default function SectionCardJ({ data }: { data: SectionJ }): React.ReactElement {
  const conf = data.confidence === 'high' || data.confidence === 'medium' || data.confidence === 'low'
    ? data.confidence
    : 'medium'

  const grouped: Record<string, RiskFinding[]> = { high: [], medium: [], low: [] }
  for (const f of data.findings ?? []) {
    const sev = f.severity === 'high' || f.severity === 'medium' || f.severity === 'low'
      ? f.severity
      : 'low'
    grouped[sev].push(f)
  }

  return (
    <SectionCard
      sectionId="J"
      sectionName="Risk & Complexity"
      confidence={conf}
      evidenceCount={data.evidence_files?.length ?? 0}
    >
      {data.summary ? (
        <p className="text-xs text-zinc-300 italic mb-3 leading-relaxed">{data.summary}</p>
      ) : null}
      <div className="space-y-3">
        {(['high', 'medium', 'low'] as const).map((sev) => {
          const items = grouped[sev]
          if (items.length === 0)
            return null
          return (
            <div key={sev}>
              <div
                className={`mb-1 text-[11px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded inline-flex items-center gap-1.5 ${SEVERITY_COLORS[sev]}`}
              >
                <span className={`inline-block w-1.5 h-1.5 rounded-full ${SEVERITY_DOT[sev]}`} />
                {sev} ({items.length})
              </div>
              <div className="space-y-1.5 mt-1">
                {items.map((f, i) => (
                  <div key={i} className={`rounded border p-2 text-xs ${SEVERITY_COLORS[sev]}`}>
                    <div className="flex flex-wrap items-baseline gap-2">
                      <span className="font-medium text-zinc-100">{f.title}</span>
                      <span className="text-[10px] uppercase font-mono px-1 py-0.5 rounded bg-zinc-950/50">
                        {f.category}
                      </span>
                    </div>
                    <div className="mt-1 text-zinc-400 leading-relaxed">{f.rationale}</div>
                    {f.evidence && f.evidence.length > 0 && (
                      <div className="mt-1 flex flex-wrap gap-1">
                        {f.evidence.slice(0, 8).map((ev, j) => (
                          <span
                            key={j}
                            className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-300"
                          >
                            {ev}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </SectionCard>
  )
}
