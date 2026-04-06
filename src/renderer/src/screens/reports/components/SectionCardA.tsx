import React from 'react'
import type { SectionA } from '../../../types/analysis'
import SectionCard from './SectionCard'

const RUNTIME_COLORS: Record<string, string> = {
  web_app: 'bg-blue-900/50 text-blue-300',
  backend_service: 'bg-purple-900/50 text-purple-300',
  monolith: 'bg-orange-900/50 text-orange-300',
  monorepo: 'bg-amber-900/40 text-amber-200',
  library: 'bg-green-900/50 text-green-300',
  cli: 'bg-zinc-800 text-zinc-300',
  worker: 'bg-slate-800 text-slate-300',
  cron: 'bg-slate-800 text-slate-300',
  unknown: 'bg-zinc-800 text-zinc-400',
}

export default function SectionCardA({ data }: { data: SectionA }): React.ReactElement {
  const rt = String(data.runtime_type ?? 'unknown')
  const rtClass = RUNTIME_COLORS[rt] ?? RUNTIME_COLORS.unknown
  const conf = data.confidence === 'high' || data.confidence === 'medium' || data.confidence === 'low'
    ? data.confidence
    : 'medium'

  return (
    <SectionCard
      sectionId="A"
      sectionName="Project Identity"
      confidence={conf}
      evidenceCount={data.evidence_files?.length ?? 0}
    >
      <div className="space-y-3">
        <h4 className="text-base font-semibold text-zinc-100">{data.repo_name}</h4>
        <div className="flex flex-wrap gap-2 items-center">
          <span className={`text-[10px] uppercase tracking-wide px-2 py-0.5 rounded ${rtClass}`}>
            {rt.replace(/_/g, ' ')}
          </span>
          {data.tech_stack?.map((t) => (
            <span key={t} className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 font-mono">
              {t}
            </span>
          ))}
        </div>
        {data.purpose ? (
          <p className="text-sm text-zinc-300 leading-relaxed">{data.purpose}</p>
        ) : null}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
          <div>
            <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-0.5">Domain</div>
            <div className="text-zinc-300">{data.domain || '—'}</div>
          </div>
          <div className="sm:col-span-2">
            <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-0.5">Business context</div>
            <div className="text-zinc-400 leading-relaxed">{data.business_context || '—'}</div>
          </div>
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
