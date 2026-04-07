import React, { useState } from 'react'

export interface SectionCardProps {
  sectionId: string
  sectionName: string
  confidence: 'high' | 'medium' | 'low'
  evidenceCount?: number
  children: React.ReactNode
  defaultCollapsed?: boolean
}

const BADGE_COLORS: Record<string, string> = {
  A: 'bg-indigo-900/50 text-indigo-300 border-indigo-800',
  B: 'bg-sky-900/50 text-sky-300 border-sky-800',
  C: 'bg-teal-900/50 text-teal-300 border-teal-800',
  D: 'bg-green-900/50 text-green-300 border-green-800',
  E: 'bg-red-900/40 text-red-300 border-red-800',
  F: 'bg-purple-900/50 text-purple-300 border-purple-800',
  G: 'bg-violet-900/50 text-violet-300 border-violet-800',
  H: 'bg-amber-900/40 text-amber-200 border-amber-800',
  I: 'bg-cyan-900/50 text-cyan-300 border-cyan-800',
  J: 'bg-rose-900/40 text-rose-300 border-rose-800',
  K: 'bg-orange-900/50 text-orange-300 border-orange-800',
}

const CONF_DOT: Record<string, string> = {
  high: 'bg-emerald-500',
  medium: 'bg-yellow-500',
  low: 'bg-zinc-500',
}

export default function SectionCard({
  sectionId,
  sectionName,
  confidence,
  evidenceCount,
  children,
  defaultCollapsed = false,
}: SectionCardProps): React.ReactElement {
  const [collapsed, setCollapsed] = useState(defaultCollapsed)
  const badge = BADGE_COLORS[sectionId] ?? 'bg-zinc-800 text-zinc-300 border-zinc-700'

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950/80 mb-3 overflow-hidden">
      <div className="flex items-center justify-between gap-2 px-3 py-2 border-b border-zinc-800/80">
        <div className="flex items-center gap-2 min-w-0">
          <span
            className={`shrink-0 rounded border px-1.5 py-0.5 text-xs font-mono font-semibold ${badge}`}
          >
            {sectionId}
          </span>
          <span className="text-sm font-semibold text-zinc-200 truncate">{sectionName}</span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span
            className="inline-flex items-center gap-1 text-[10px] uppercase tracking-wide text-zinc-400"
            title={`Confidence: ${confidence}`}
          >
            <span className={`inline-block w-1.5 h-1.5 rounded-full ${CONF_DOT[confidence] ?? 'bg-zinc-500'}`} />
            {confidence}
          </span>
          {evidenceCount !== undefined && evidenceCount > 0 && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 font-mono">
              {evidenceCount} ev
            </span>
          )}
          <button
            type="button"
            onClick={() => setCollapsed((c) => !c)}
            className="p-1 rounded border border-zinc-700 text-zinc-400 hover:border-zinc-600 hover:text-zinc-200"
            aria-expanded={!collapsed}
            aria-label={collapsed ? 'Expand section' : 'Collapse section'}
          >
            <span className="text-xs leading-none">{collapsed ? '▸' : '▾'}</span>
          </button>
        </div>
      </div>
      {!collapsed && <div className="px-3 py-3">{children}</div>}
    </div>
  )
}
