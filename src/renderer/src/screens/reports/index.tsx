import React, { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import type { AnalysisReport, AnalysisReportSummary } from '../../types/electron'
import { ErrorBanner } from '../../components/ui/ErrorBanner'
import { toErrorMessage } from '../../lib/errors'

export default function ReportViewerScreen(): React.ReactElement {
  const [params] = useSearchParams()
  const repoId = params.get('repoId') ?? undefined
  const reportIdInUrl = params.get('reportId') ?? ''

  const [reports, setReports] = useState<AnalysisReportSummary[]>([])
  const [selectedReportId, setSelectedReportId] = useState('')
  const [report, setReport] = useState<AnalysisReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const run = async () => {
      setLoading(true)
      setError(null)
      try {
        const list = await window.api.analysis.listReports(repoId, 50)
        setReports(list)
        setSelectedReportId(reportIdInUrl || list[0]?.id || '')
      } catch (err) {
        setError(toErrorMessage(err))
      } finally {
        setLoading(false)
      }
    }
    run()
  }, [repoId, reportIdInUrl])

  useEffect(() => {
    const run = async () => {
      if (!selectedReportId) {
        setReport(null)
        return
      }
      try {
        const out = await window.api.analysis.getReport(selectedReportId)
        setReport(out)
      } catch (err) {
        setError(toErrorMessage(err))
      }
    }
    run()
  }, [selectedReportId])

  const sections = useMemo(() => report?.report.sections ?? [], [report])
  const confidence = report?.report.confidence_summary

  return (
    <>
      <div className="screen-header">
        <h1 className="screen-title">Reports</h1>
        <p className="screen-subtitle">View generated analysis artifacts</p>
      </div>
      <div className="h-[calc(100vh-10rem)] overflow-y-auto p-4 space-y-3">
        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
          <div className="bg-zinc-900/60 border border-zinc-700 rounded-xl p-3 space-y-2">
            <div className="text-xs font-semibold text-zinc-100">Reports</div>
            {loading ? (
              <div className="text-xs text-zinc-500">Loading...</div>
            ) : reports.length === 0 ? (
              <div className="text-xs text-zinc-500">No report yet. Run analysis first.</div>
            ) : (
              <div className="space-y-1.5 max-h-[68vh] overflow-auto">
                {reports.map((r) => (
                  <button
                    key={r.id}
                    onClick={() => setSelectedReportId(r.id)}
                    className={`w-full text-left rounded-md border px-2 py-1.5 text-[11px] ${
                      r.id === selectedReportId
                        ? 'border-indigo-600 bg-indigo-950/40 text-indigo-200'
                        : 'border-zinc-700 bg-zinc-950 text-zinc-300 hover:border-zinc-600'
                    }`}
                  >
                    <div className="font-mono truncate">{r.id.slice(0, 10)} · {r.model_id}</div>
                    <div className="text-zinc-500 truncate">{r.created_at}</div>
                    <div className="text-zinc-500 truncate">{r.scan_mode} · {r.privacy_mode}</div>
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="lg:col-span-2 bg-zinc-900/60 border border-zinc-700 rounded-xl p-3 space-y-3">
            {report ? (
              <>
                <div className="text-xs text-zinc-400">
                  job: <span className="text-zinc-200 font-mono">{report.summary.job_id}</span>
                  <span className="mx-2 text-zinc-700">|</span>
                  repo: <span className="text-zinc-200 font-mono">{report.summary.repo_id}</span>
                  <span className="mx-2 text-zinc-700">|</span>
                  snapshot: <span className="text-zinc-200 font-mono">{report.summary.snapshot_id}</span>
                </div>
                {confidence && (
                  <div className="text-xs text-zinc-400">
                    confidence - high: <span className="text-zinc-200">{confidence.high}</span>
                    <span className="mx-2 text-zinc-700">|</span>
                    medium: <span className="text-zinc-200">{confidence.medium}</span>
                    <span className="mx-2 text-zinc-700">|</span>
                    low: <span className="text-zinc-200">{confidence.low}</span>
                  </div>
                )}
                <div className="space-y-2 max-h-[64vh] overflow-auto pr-1">
                  {sections.map((s, i) => (
                    <div key={`${s.section}-${i}`} className="rounded-md border border-zinc-800 bg-zinc-950 p-2 space-y-1">
                      <div className="text-xs text-zinc-200">
                        {s.section} · <span className="text-zinc-500">{s.confidence}</span>
                      </div>
                      <div className="text-xs text-zinc-400">{s.content}</div>
                      {s.evidence_files.length > 0 && (
                        <div className="text-[11px] text-zinc-500">
                          evidence: {s.evidence_files.slice(0, 8).join(', ')}
                        </div>
                      )}
                      {s.blind_spots.length > 0 && (
                        <div className="text-[11px] text-amber-400">
                          blind spots: {s.blind_spots.join(', ')}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="text-xs text-zinc-500">Select a report to view details.</div>
            )}
          </div>
        </div>
      </div>
    </>
  )
}
