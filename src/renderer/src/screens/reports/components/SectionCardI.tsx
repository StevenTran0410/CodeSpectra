import React, { useMemo, useState } from 'react'
import type { SectionI } from '../../../types/analysis'
import { normConf } from '../../../lib/reportUtils'
import SectionCard, { type SectionCardRerunProps } from './SectionCard'

export default function SectionCardI({
  data,
  onRerun,
  rerunBusy,
}: { data: SectionI } & SectionCardRerunProps): React.ReactElement {
  const [query, setQuery] = useState('')
  const conf = normConf(data.confidence)

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    const terms = data.terms ?? []
    if (!q)
      return terms
    return terms.filter(
      (t) =>
        t.term.toLowerCase().includes(q) ||
        t.definition.toLowerCase().includes(q) ||
        (t.evidence_files ?? []).some((f) => f.toLowerCase().includes(q)),
    )
  }, [data.terms, query])

  return (
    <SectionCard
      sectionId="I"
      sectionName="Domain Glossary"
      confidence={conf}
      onRerun={onRerun}
      rerunBusy={rerunBusy}
    >
      <div className="space-y-2">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Filter terms…"
          className="w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-200 placeholder:text-zinc-600"
        />
        {filtered.length === 0 ? (
          <div className="text-xs text-zinc-500 italic py-2">No terms match.</div>
        ) : (
          <div className="border border-zinc-800 rounded overflow-hidden">
            {filtered.map((t, idx) => (
              <div
                key={`${t.term}-${idx}`}
                className="grid grid-cols-1 sm:grid-cols-[minmax(0,140px)_1fr] gap-2 border-b border-zinc-800 last:border-b-0 px-2 py-2"
              >
                <div className="font-mono text-sm font-medium text-zinc-100 break-words">{t.term}</div>
                <div>
                  <div className="text-sm text-zinc-400 leading-relaxed">{t.definition}</div>
                  {(t.evidence_files?.length ?? 0) > 0 && (
                    <div className="flex flex-wrap gap-1 mt-1">
                      {t.evidence_files!.map((f, j) => (
                        <span
                          key={j}
                          className="font-mono text-[10px] bg-zinc-800 text-zinc-400 rounded px-1 py-0.5 break-all"
                        >
                          {f}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </SectionCard>
  )
}
