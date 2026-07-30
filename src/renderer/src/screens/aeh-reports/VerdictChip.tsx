import React from 'react'
import { AlertTriangle, Bot, CheckCircle2, HelpCircle } from 'lucide-react'
import { EvalCaseItem, AgentVerdictInfo } from '../../store/aeh-reports.store'

export function computeClientAgentVerdict(
  cases: EvalCaseItem[],
  threshold: number = 0.70
): AgentVerdictInfo {
  const lowCases = cases.filter((c) => {
    const sm = c.evaluations?.find((e) => e.metric_name === 'semantic_match')
    return typeof sm?.score === 'number' && sm.score < threshold
  })

  if (lowCases.length === 0) {
    return { verdict: 'healthy', d_count: 0, a_count: 0, low_count: 0 }
  }

  let dCount = 0
  let aCount = 0
  for (const c of lowCases) {
    const flags = c.flags
    const hasDatasetFlag =
      flags?.expected_suspect ||
      flags?.input_starved ||
      (flags?.schema_suspect_fields && flags.schema_suspect_fields.length > 0)
    if (hasDatasetFlag) {
      dCount++
    } else {
      aCount++
    }
  }

  const dRatio = dCount / lowCases.length
  const aRatio = aCount / lowCases.length

  let verdict: 'healthy' | 'likely-dataset' | 'likely-agent' | 'mixed' = 'mixed'
  if (dRatio >= 0.6) verdict = 'likely-dataset'
  else if (aRatio >= 0.6) verdict = 'likely-agent'

  return { verdict, d_count: dCount, a_count: aCount, low_count: lowCases.length }
}

export function VerdictChip({
  verdictInfo,
  compact = false
}: {
  verdictInfo: AgentVerdictInfo
  compact?: boolean
}): React.ReactElement {
  const { verdict, d_count, a_count, low_count } = verdictInfo

  if (verdict === 'healthy') {
    return (
      <span
        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-950/40 text-emerald-300 border border-emerald-800/50"
        title="All cases scored above acceptable threshold"
      >
        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 flex-shrink-0" />
        <span>healthy {compact ? '' : '· 0 low cases'}</span>
      </span>
    )
  }

  if (verdict === 'likely-dataset') {
    return (
      <span
        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-950/40 text-amber-300 border border-amber-800/50"
        title={`${d_count} of ${low_count} low cases have suspect gold, starved input, or schema mismatches`}
      >
        <AlertTriangle className="h-3.5 w-3.5 text-amber-400 flex-shrink-0" />
        <span>
          likely dataset {compact ? `(${d_count}/${low_count})` : `· ${d_count} of ${low_count} low cases (${d_count} dataset / ${a_count} agent)`}
        </span>
      </span>
    )
  }

  if (verdict === 'likely-agent') {
    return (
      <span
        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-rose-950/40 text-rose-300 border border-rose-800/50"
        title={`${a_count} of ${low_count} low cases have genuine agent omissions or errors`}
      >
        <Bot className="h-3.5 w-3.5 text-rose-400 flex-shrink-0" />
        <span>
          likely agent {compact ? `(${a_count}/${low_count})` : `· ${a_count} of ${low_count} low cases (${d_count} dataset / ${a_count} agent)`}
        </span>
      </span>
    )
  }

  return (
    <span
      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-purple-950/40 text-purple-300 border border-purple-800/50"
      title={`Mixed attribution: ${d_count} dataset vs ${a_count} agent low cases`}
    >
      <HelpCircle className="h-3.5 w-3.5 text-purple-400 flex-shrink-0" />
      <span>
        mixed {compact ? `(${d_count}D/${a_count}A)` : `· ${low_count} low cases (${d_count} dataset / ${a_count} agent)`}
      </span>
    </span>
  )
}
