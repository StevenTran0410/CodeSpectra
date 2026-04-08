import React from 'react'
import type { Confidence, SectionL } from '../../../types/analysis'
import SectionCard, { type SectionCardRerunProps } from './SectionCard'

function normConf(c: string | undefined): Confidence {
  return c === 'high' || c === 'medium' || c === 'low' ? c : 'medium'
}

const PROSE_FIELDS: { key: keyof SectionL; label: string }[] = [
  { key: 'executive_summary', label: 'Executive Summary' },
  { key: 'architecture_narrative', label: 'Architecture' },
  { key: 'tech_stack_snapshot', label: 'Tech Stack' },
  { key: 'developer_quickstart', label: 'Developer Quickstart' },
  { key: 'conventions_digest', label: 'Conventions' },
  { key: 'risk_highlights', label: 'Risk Highlights' },
  { key: 'reading_path', label: 'Reading Path' },
]

export default function SectionCardL({
  data,
  onRerun,
  rerunBusy,
}: { data: SectionL } & SectionCardRerunProps): React.ReactElement {
  const conf = normConf(data.confidence)

  return (
    <SectionCard
      sectionId="L"
      sectionName="Synthesis Report"
      confidence={conf}
      onRerun={onRerun}
      rerunBusy={rerunBusy}
    >
      <div className="space-y-3">
        {PROSE_FIELDS.map(({ key, label }) => {
          const value = data[key] as string
          if (!value) return null
          return (
            <div key={key}>
              <div className="text-xs uppercase text-zinc-500 mb-1">{label}</div>
              <p className="text-sm text-zinc-300 whitespace-pre-wrap leading-relaxed">{value}</p>
            </div>
          )
        })}
      </div>
    </SectionCard>
  )
}
