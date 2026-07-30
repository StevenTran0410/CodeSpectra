import React, { useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  TrendingUp,
  TrendingDown,
  Minus,
  AlertTriangle,
  ArrowLeft,
  GitCompare
} from 'lucide-react'
import { useAEHReportsStore, EvalCaseItem } from '../../store/aeh-reports.store'

function _scoreColor(score: number | null): string {
  if (score === null) return 'text-slate-400'
  if (score >= 0.7) return 'text-emerald-400'
  if (score >= 0.4) return 'text-amber-400'
  return 'text-rose-400'
}

function formatDelta(delta: number | null): React.ReactElement {
  if (delta === null || isNaN(delta)) return <span className="text-gray-500 font-mono">N/A</span>
  const valStr = (delta > 0 ? '+' : '') + (delta * 100).toFixed(1) + '%'
  if (Math.abs(delta) < 0.001) {
    return (
      <span className="inline-flex items-center gap-1 text-slate-400 font-mono font-bold">
        <Minus className="h-3.5 w-3.5" />
        <span>0.0%</span>
      </span>
    )
  }
  if (delta > 0) {
    return (
      <span className="inline-flex items-center gap-1 text-emerald-400 font-mono font-bold">
        <TrendingUp className="h-3.5 w-3.5" />
        <span>{valStr}</span>
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1 text-rose-400 font-mono font-bold">
      <TrendingDown className="h-3.5 w-3.5" />
      <span>{valStr}</span>
    </span>
  )
}

function getCaseScore(caseItem: EvalCaseItem | undefined): number | null {
  if (!caseItem) return null
  const sm = caseItem.evaluations?.find((e) => e.metric_name === 'semantic_match')
  return typeof sm?.score === 'number' ? sm.score : null
}

export default function RunCompareScreen(): React.ReactElement {
  const { runIdA, runIdB } = useParams<{ runIdA: string; runIdB: string }>()
  const navigate = useNavigate()

  const {
    compareCasesA,
    compareCasesB,
    compareRunItemA,
    compareRunItemB,
    loading,
    error,
    fetchComparison
  } = useAEHReportsStore()

  useEffect(() => {
    if (runIdA && runIdB) {
      fetchComparison(runIdA, runIdB).catch(() => {})
    }
  }, [runIdA, runIdB, fetchComparison])

  if (loading && (!compareCasesA || !compareCasesB)) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center py-20 gap-4 text-slate-100 bg-slate-950">
        <div className="h-10 w-10 rounded-full border-t-2 border-indigo-500 animate-spin"></div>
        <p className="text-sm text-gray-400 font-medium">Loading run comparison...</p>
      </div>
    )
  }

  if (error || !compareCasesA || !compareCasesB || !runIdA || !runIdB) {
    return (
      <div className="flex-1 p-6 text-slate-100 bg-slate-950">
        <button
          onClick={() => navigate('/aeh/reports')}
          className="mb-4 px-4 py-2 text-xs font-semibold text-gray-300 hover:text-white bg-slate-900 border border-slate-800 rounded-lg transition-colors flex items-center gap-2 cursor-pointer"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Runs
        </button>
        <div className="p-4 rounded-xl border border-rose-900/40 bg-rose-950/20 text-rose-300 flex items-center gap-3 text-sm font-medium">
          <AlertTriangle className="h-5 w-5 flex-shrink-0 text-rose-400" />
          <p>{error || 'Comparison details could not be loaded'}</p>
        </div>
      </div>
    )
  }

  const scoreA = compareRunItemA?.avg_semantic_match ?? null
  const scoreB = compareRunItemB?.avg_semantic_match ?? null
  const overallDelta =
    scoreA !== null && scoreB !== null ? scoreB - scoreA : null

  // Collect all unique dataset_case_ids for pairing
  const allCasesA: EvalCaseItem[] = []
  Object.values(compareCasesA.agents || {}).forEach((list) => allCasesA.push(...list))

  const allCasesB: EvalCaseItem[] = []
  Object.values(compareCasesB.agents || {}).forEach((list) => allCasesB.push(...list))

  const casesByDatasetCaseId: Record<
    string,
    { caseA?: EvalCaseItem; caseB?: EvalCaseItem; agentId: string }
  > = {}

  allCasesA.forEach((ca) => {
    const key = ca.case_id || ca.trace_id
    casesByDatasetCaseId[key] = { caseA: ca, agentId: 'unknown' }
  })

  allCasesB.forEach((cb) => {
    const key = cb.case_id || cb.trace_id
    if (!casesByDatasetCaseId[key]) {
      casesByDatasetCaseId[key] = { caseB: cb, agentId: 'unknown' }
    } else {
      casesByDatasetCaseId[key].caseB = cb
    }
  })

  // Agent-level comparison table
  const allAgentIds = Array.from(
    new Set([
      ...Object.keys(compareCasesA.agents || {}),
      ...Object.keys(compareCasesB.agents || {})
    ])
  )

  const agentRows = allAgentIds.map((agentId) => {
    const casesA = compareCasesA.agents[agentId] || []
    const casesB = compareCasesB.agents[agentId] || []

    const smA = casesA
      .map((c) => getCaseScore(c))
      .filter((s): s is number => s !== null)
    const smB = casesB
      .map((c) => getCaseScore(c))
      .filter((s): s is number => s !== null)

    const avgA = smA.length > 0 ? smA.reduce((a, b) => a + b, 0) / smA.length : null
    const avgB = smB.length > 0 ? smB.reduce((a, b) => a + b, 0) / smB.length : null
    const delta = avgA !== null && avgB !== null ? avgB - avgA : null

    return {
      agentId,
      caseCountA: casesA.length,
      caseCountB: casesB.length,
      avgA,
      avgB,
      delta
    }
  })

  return (
    <div className="flex-1 flex flex-col min-h-screen text-slate-100 bg-slate-950 pb-10">
      {/* Navigation Header */}
      <header className="border-b border-slate-800 bg-slate-900 px-6 py-4 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center">
            <GitCompare className="h-5 w-5 text-indigo-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-gray-100">Run Comparison</h1>
            <p className="text-xs text-gray-400">
              Baseline ({runIdA.slice(0, 8)}) vs Target ({runIdB.slice(0, 8)})
            </p>
          </div>
        </div>

        <button
          onClick={() => navigate('/aeh/reports')}
          className="px-4 py-2 text-xs font-semibold text-gray-300 hover:text-white bg-slate-800 border border-slate-700 rounded-lg transition-colors flex items-center gap-1.5 cursor-pointer"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Back to Runs
        </button>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 flex flex-col gap-6">
        {/* Comparison Hero Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
          {/* Baseline Run A */}
          <div className="flex flex-col gap-1">
            <span className="text-xs font-bold text-gray-400 uppercase tracking-wider">
              Baseline Run A
            </span>
            <span className="font-mono text-base font-bold text-gray-100">
              {compareRunItemA?.system_name || runIdA.slice(0, 8)}
            </span>
            <div className="flex items-baseline gap-2 mt-1">
              <span className={`text-3xl font-extrabold ${_scoreColor(scoreA)}`}>
                {scoreA !== null ? `${(scoreA * 100).toFixed(0)}%` : 'N/A'}
              </span>
              <span className="text-xs text-gray-400">avg semantic match</span>
            </div>
          </div>

          {/* Delta Indicator */}
          <div className="flex flex-col items-center justify-center border-y md:border-y-0 md:border-x border-slate-800/80 py-4 md:py-0 md:px-6">
            <span className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">
              Overall Semantic Match Delta
            </span>
            <div className="text-2xl font-black">{formatDelta(overallDelta)}</div>
          </div>

          {/* Target Run B */}
          <div className="flex flex-col gap-1 text-right">
            <span className="text-xs font-bold text-gray-400 uppercase tracking-wider">
              Target Run B
            </span>
            <span className="font-mono text-base font-bold text-gray-100">
              {compareRunItemB?.system_name || runIdB.slice(0, 8)}
            </span>
            <div className="flex items-baseline justify-end gap-2 mt-1">
              <span className={`text-3xl font-extrabold ${_scoreColor(scoreB)}`}>
                {scoreB !== null ? `${(scoreB * 100).toFixed(0)}%` : 'N/A'}
              </span>
              <span className="text-xs text-gray-400">avg semantic match</span>
            </div>
          </div>
        </div>

        {/* Per-Agent Comparison Table */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden flex flex-col">
          <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/80">
            <h3 className="text-sm font-bold text-gray-100">Per-Agent Performance Delta</h3>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-800 bg-slate-950/60 text-gray-400 text-xs font-semibold uppercase tracking-wider">
                  <th className="py-3 px-6">Agent ID</th>
                  <th className="py-3 px-6 text-center">Baseline Run A</th>
                  <th className="py-3 px-6 text-center">Target Run B</th>
                  <th className="py-3 px-6 text-center">Score Delta</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-sm">
                {agentRows.map((row) => (
                  <tr key={row.agentId} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-3.5 px-6 font-mono font-bold text-gray-100">
                      {row.agentId}
                    </td>
                    <td className="py-3.5 px-6 text-center font-mono font-semibold">
                      <span className={_scoreColor(row.avgA)}>
                        {row.avgA !== null ? `${(row.avgA * 100).toFixed(0)}%` : 'N/A'}
                      </span>
                    </td>
                    <td className="py-3.5 px-6 text-center font-mono font-semibold">
                      <span className={_scoreColor(row.avgB)}>
                        {row.avgB !== null ? `${(row.avgB * 100).toFixed(0)}%` : 'N/A'}
                      </span>
                    </td>
                    <td className="py-3.5 px-6 text-center">{formatDelta(row.delta)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Per-Case Comparison Breakdown */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden flex flex-col">
          <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/80">
            <h3 className="text-sm font-bold text-gray-100">
              Per-Case Delta Breakdown ({Object.keys(casesByDatasetCaseId).length} Cases)
            </h3>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-800 bg-slate-950/60 text-gray-400 text-xs font-semibold uppercase tracking-wider">
                  <th className="py-3 px-6">Case ID</th>
                  <th className="py-3 px-6 text-center">Baseline Score A</th>
                  <th className="py-3 px-6 text-center">Target Score B</th>
                  <th className="py-3 px-6 text-center">Delta</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-sm">
                {Object.entries(casesByDatasetCaseId).map(([caseKey, item]) => {
                  const sA = getCaseScore(item.caseA)
                  const sB = getCaseScore(item.caseB)
                  const delta = sA !== null && sB !== null ? sB - sA : null
                  return (
                    <tr key={caseKey} className="hover:bg-slate-800/40 transition-colors">
                      <td className="py-3 px-6 font-mono font-semibold text-gray-200 text-xs">
                        {caseKey}
                      </td>
                      <td className="py-3 px-6 text-center font-mono font-semibold text-xs">
                        <span className={_scoreColor(sA)}>
                          {sA !== null ? `${(sA * 100).toFixed(0)}%` : 'N/A'}
                        </span>
                      </td>
                      <td className="py-3 px-6 text-center font-mono font-semibold text-xs">
                        <span className={_scoreColor(sB)}>
                          {sB !== null ? `${(sB * 100).toFixed(0)}%` : 'N/A'}
                        </span>
                      </td>
                      <td className="py-3 px-6 text-center text-xs">{formatDelta(delta)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  )
}
