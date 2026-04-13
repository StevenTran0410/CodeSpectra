import React, { useState } from 'react'
import { FileText, ChevronDown, ChevronUp } from 'lucide-react'
import type { SectionA } from '../../../types/analysis'
import { normConf } from '../../../lib/reportUtils'
import SectionCard, { type SectionCardRerunProps } from './SectionCard'

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

export default function SectionCardA({
  data,
  onRerun,
  rerunBusy,
  featureNames = [],
  entrypoints = [],
}: { data: SectionA; featureNames?: string[]; entrypoints?: string[] } & SectionCardRerunProps): React.ReactElement {
  const rt = String(data.runtime_type ?? 'unknown')
  const rtClass = RUNTIME_COLORS[rt] ?? RUNTIME_COLORS.unknown
  const conf = normConf(data.confidence)
  const [filesOpen, setFilesOpen] = useState(false)
  const evidenceFiles = data.evidence_files ?? []

  return (
    <SectionCard
      sectionId="A"
      sectionName="Project Identity"
      confidence={conf}
      evidenceCount={evidenceFiles.length}
      onRerun={onRerun}
      rerunBusy={rerunBusy}
    >
      <div className="space-y-3">
        <h4 className="text-base font-semibold text-zinc-100">{data.repo_name}</h4>

        {/* Runtime type + tech stack */}
        <div className="flex flex-wrap gap-1.5 items-center">
          <span className={`text-[10px] uppercase tracking-wide px-2 py-0.5 rounded font-semibold ${rtClass}`}>
            {rt.replace(/_/g, ' ')}
          </span>
          {data.tech_stack?.map((t) => (
            <span key={t} className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 font-mono">
              {t}
            </span>
          ))}
        </div>

        {/* Purpose */}
        {data.purpose && (
          <p className="text-sm text-zinc-300 leading-relaxed">{data.purpose}</p>
        )}

        {/* Meta grid: domain, runtime env, entrypoints */}
        <div className="grid grid-cols-1 gap-2 text-xs">
          {data.domain && (
            <div className="flex gap-2 items-baseline">
              <span className="text-[10px] uppercase tracking-wide text-zinc-500 shrink-0 w-24">Domain</span>
              <span className="text-zinc-300">{data.domain}</span>
            </div>
          )}
          {entrypoints.length > 0 && (
            <div className="flex gap-2 items-start">
              <span className="text-[10px] uppercase tracking-wide text-zinc-500 shrink-0 w-24 mt-0.5">Entrypoints</span>
              <div className="flex flex-col gap-0.5 min-w-0">
                {entrypoints.slice(0, 4).map((ep) => (
                  <span key={ep} className="font-mono text-[10px] text-zinc-300 break-all">{ep}</span>
                ))}
              </div>
            </div>
          )}
          {data.business_context && (
            <div>
              <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-0.5">Business context</div>
              <div className="text-zinc-400 leading-relaxed">{data.business_context}</div>
            </div>
          )}
        </div>

        {/* Key capabilities from Section F */}
        {featureNames.length > 0 && (
          <div>
            <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-1">Key capabilities</div>
            <div className="flex flex-wrap gap-1.5">
              {featureNames.map((name) => (
                <span
                  key={name}
                  className="text-[10px] px-2 py-0.5 rounded-full bg-zinc-800/80 border border-zinc-700/60 text-zinc-300"
                >
                  {name}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Evidence files (collapsible) */}
        {evidenceFiles.length > 0 && (
          <div className="rounded border border-zinc-800 overflow-hidden">
            <button
              type="button"
              onClick={() => setFilesOpen((v) => !v)}
              className="w-full flex items-center justify-between px-2 py-1.5 text-left hover:bg-zinc-900/40 transition-colors"
            >
              <span className="text-[10px] uppercase tracking-wide text-zinc-500">
                Analyzed from {evidenceFiles.length} file{evidenceFiles.length !== 1 ? 's' : ''}
              </span>
              {filesOpen
                ? <ChevronUp size={12} className="text-zinc-600" />
                : <ChevronDown size={12} className="text-zinc-600" />
              }
            </button>
            {filesOpen && (
              <ul className="border-t border-zinc-800 px-2 py-1.5 space-y-0.5 max-h-48 overflow-y-auto">
                {evidenceFiles.map((f) => (
                  <li key={f} className="flex items-start gap-1 font-mono text-[10px] text-zinc-400">
                    <FileText size={10} className="text-zinc-600 shrink-0 mt-0.5" />
                    <span className="break-all">{f}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {/* Blind spots */}
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
