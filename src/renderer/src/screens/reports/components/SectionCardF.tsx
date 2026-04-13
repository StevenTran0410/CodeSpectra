import React, { useState } from 'react'
import { Beaker, FileText, Maximize2 } from 'lucide-react'
import type { FeatureMapItem, SectionF } from '../../../types/analysis'
import { normConf } from '../../../lib/reportUtils'
import C4DiagramView from '../../../components/C4DiagramView'
import DiagramModal from '../../../components/DiagramModal'
import SequenceFlowView from '../../../components/SequenceFlowView'
import SectionCard, { type SectionCardRerunProps } from './SectionCard'

export default function SectionCardF({
  data,
  onRerun,
  rerunBusy,
}: { data: SectionF } & SectionCardRerunProps): React.ReactElement {
  const conf = normConf(data.confidence)
  const hasDiagram = !!data.mermaid_diagram
  const [expanded, setExpanded] = useState<Set<number>>(() => new Set())
  const [activeTab, setActiveTab] = useState<'details' | 'diagram'>('details')
  const [diagramOpen, setDiagramOpen] = useState(false)

  const toggle = (idx: number) => {
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }

  return (
    <SectionCard
      sectionId="F"
      sectionName="Feature Map"
      confidence={conf}
      evidenceCount={data.evidence_files?.length ?? 0}
      onRerun={onRerun}
      rerunBusy={rerunBusy}
    >
      {hasDiagram && (
        <div className="flex items-center gap-2 mb-3 border-b border-zinc-800 pb-2">
          {(['details', 'diagram'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`text-[10px] uppercase tracking-wide px-2 py-0.5 rounded ${
                activeTab === tab
                  ? 'bg-zinc-700 text-zinc-100'
                  : 'text-zinc-500 hover:text-zinc-300'
              }`}
            >
              {tab}
            </button>
          ))}
          {activeTab === 'diagram' && (
            <button
              onClick={() => setDiagramOpen(true)}
              className="ml-auto flex items-center gap-1 text-[10px] text-zinc-500 hover:text-zinc-300 transition-colors"
              title="Open fullscreen"
            >
              <Maximize2 size={12} />
            </button>
          )}
        </div>
      )}
      {hasDiagram && data.mermaid_diagram && (
        <DiagramModal
          open={diagramOpen}
          onClose={() => setDiagramOpen(false)}
          diagramData={data.mermaid_diagram}
          title="Feature Map"
        />
      )}
      {activeTab === 'diagram' && data.mermaid_diagram ? (
        <C4DiagramView data={data.mermaid_diagram} />
      ) : (
      <div className="space-y-3">
        {!data.features || data.features.length === 0 ? (
          <div className="text-zinc-500 italic text-xs">No features detected</div>
        ) : (
          <div className="space-y-0">
            {data.features.map((feat: FeatureMapItem, idx: number) => {
              const open = expanded.has(idx)
              return (
                <div key={idx} className="border border-zinc-800 rounded-md mb-2 overflow-hidden">
                  <button
                    type="button"
                    className="w-full flex items-center justify-between gap-2 px-2 py-2 text-left cursor-pointer hover:bg-zinc-900/40"
                    onClick={() => toggle(idx)}
                  >
                    <div className="min-w-0 flex items-baseline gap-2 flex-wrap">
                      <span className="text-zinc-100 text-sm font-semibold">
                        {feat.name || '—'}
                      </span>
                      <span className="font-mono text-[10px] text-zinc-400 truncate">
                        {feat.entrypoint || ''}
                      </span>
                    </div>
                    <span className="text-zinc-500 shrink-0">{open ? '▾' : '▸'}</span>
                  </button>
                  {open && (
                    <div className="px-2 pb-2 pt-0 space-y-2 border-t border-zinc-800/80">
                      {feat.description ? (
                        <p className="text-xs text-zinc-400">{feat.description}</p>
                      ) : null}
                      {feat.key_files && feat.key_files.length > 0 && (
                        <div>
                          <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-0.5">
                            Key files
                          </div>
                          <ul className="space-y-0.5">
                            {feat.key_files.map((f) => (
                              <li
                                key={f}
                                className="font-mono text-[11px] text-zinc-300 pl-1 flex gap-1 items-start"
                              >
                                <FileText className="w-3 h-3 text-zinc-600 shrink-0 mt-0.5" />
                                <span className="break-all">{f}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      <div>
                        <span className="text-zinc-500 text-[10px]">Data</span>
                        <div className="font-mono text-[11px] text-zinc-300 break-all">
                          {feat.data_path?.trim() ? feat.data_path : '—'}
                        </div>
                      </div>
                      {feat.tests && feat.tests.length > 0 && (
                        <div>
                          <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-0.5">
                            Tests
                          </div>
                          <ul className="space-y-0.5">
                            {feat.tests.map((t) => (
                              <li
                                key={t}
                                className="font-mono text-[11px] text-zinc-400 pl-1 flex gap-1 items-start"
                              >
                                <Beaker className="w-3 h-3 text-zinc-600 shrink-0 mt-0.5" />
                                <span className="break-all">{t}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {feat.reading_order && feat.reading_order.length > 0 && (
                        <div>
                          <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-0.5">
                            Reading order
                          </div>
                          <ol className="list-decimal pl-4 space-y-0.5 marker:text-zinc-500">
                            {feat.reading_order.map((step, i) => (
                              <li key={`${i}-${step}`} className="text-xs text-zinc-300 pl-1">
                                {step}
                              </li>
                            ))}
                          </ol>
                        </div>
                      )}
                      {feat.sequence_diagram && feat.sequence_diagram.nodes.length >= 2 && (
                        <div>
                          <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-1">
                            Call flow
                          </div>
                          <SequenceFlowView data={feat.sequence_diagram} />
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
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
      )}
    </SectionCard>
  )
}
