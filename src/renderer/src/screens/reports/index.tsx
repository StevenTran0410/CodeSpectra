import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import type {
  AnalysisReport,
  AnalysisReportSummary,
  ReportDiffResult,
  StalenessResult,
} from '../../types/electron'
import {
  type SectionA,
  type SectionB,
  type SectionC,
  type SectionD,
  type SectionE,
  type SectionF,
  type SectionG,
  type SectionH,
  type SectionI,
  type SectionJ,
  type SectionK,
  type SectionL,
} from '../../types/analysis'
import { ErrorBanner } from '../../components/ui/ErrorBanner'
import { Button, ConfirmDialog, Modal, useToastStore, ErrorBoundary } from '../../components/ui'
import { toErrorMessage } from '../../lib/errors'
import EvidencePanel from '../../components/EvidencePanel'
import SectionCardA from './components/SectionCardA'
import SectionCardB from './components/SectionCardB'
import SectionCardC from './components/SectionCardC'
import SectionCardD from './components/SectionCardD'
import SectionCardE from './components/SectionCardE'
import SectionCardF from './components/SectionCardF'
import SectionCardG from './components/SectionCardG'
import SectionCardH from './components/SectionCardH'
import SectionCardI from './components/SectionCardI'
import SectionCardJ from './components/SectionCardJ'
import SectionCardK from './components/SectionCardK'
import SectionCardL from './components/SectionCardL'

function getReportSections(raw: unknown): Record<string, unknown> | null {
  if (!raw || typeof raw !== 'object') return null
  const r = raw as Record<string, unknown>
  const blob =
    r.version === 2 || r.version === 3 ? r.sections : r.sections_v2
  return blob && typeof blob === 'object' ? (blob as Record<string, unknown>) : null
}

const REPORT_SECTION_ORDER = [
  'L',
  'A',
  'B',
  'C',
  'D',
  'E',
  'F',
  'G',
  'H',
  'I',
  'J',
  'K',
] as const

const SECTION_COMPONENTS: Record<string, React.ComponentType<any>> = {
  L: SectionCardL, A: SectionCardA, B: SectionCardB, C: SectionCardC,
  D: SectionCardD, E: SectionCardE, F: SectionCardF, G: SectionCardG,
  H: SectionCardH, I: SectionCardI, J: SectionCardJ, K: SectionCardK,
}

type ExtraPropsCtx = { exportAudit: () => Promise<void>; exportAuditBusy: boolean }
const SECTION_EXTRA_PROPS: Record<string, (ctx: ExtraPropsCtx) => Record<string, unknown>> = {
  K: (ctx) => ({ onExportAudit: () => void ctx.exportAudit(), exportAuditBusy: ctx.exportAuditBusy }),
}

export default function ReportViewerScreen(): React.ReactElement {
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const toast = useToastStore()
  const repoId = params.get('repoId') ?? undefined
  const reportIdInUrl = params.get('reportId') ?? ''
  const isDetailMode = reportIdInUrl.trim().length > 0

  const [reports, setReports] = useState<AnalysisReportSummary[]>([])
  const [selectedReportId, setSelectedReportId] = useState('')
  const [report, setReport] = useState<AnalysisReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [exportingMd, setExportingMd] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [hideDeleteWarning, setHideDeleteWarning] = useState(
    localStorage.getItem('reports.deleteWarningHidden') === '1'
  )
  const [rerunLetter, setRerunLetter] = useState<string | null>(null)
  const [exportAuditBusy, setExportAuditBusy] = useState(false)
  const [repoReportsForCompare, setRepoReportsForCompare] = useState<AnalysisReportSummary[]>([])
  const [compareOpen, setCompareOpen] = useState(false)
  const [compareOtherId, setCompareOtherId] = useState('')
  const [compareBusy, setCompareBusy] = useState(false)
  const [diffResult, setDiffResult] = useState<ReportDiffResult | null>(null)
  const [sourcesPanel, setSourcesPanel] = useState<{
    sectionId: string
    sources: Array<{ chunk_id: string; rel_path: string; chunk_index: number; snippet: string }>
    loading: boolean
  } | null>(null)
  const [staleness, setStaleness] = useState<StalenessResult | null>(null)
  const [stalenessLoading, setStalenessLoading] = useState(false)

  const showSources = useCallback(async (sectionId: string) => {
    if (!report) return
    setSourcesPanel({ sectionId, sources: [], loading: true })
    try {
      const res = await window.api.analysis.getSectionSources(report.summary.id, sectionId)
      setSourcesPanel({ sectionId, sources: res.sources, loading: false })
    } catch {
      setSourcesPanel({ sectionId, sources: [], loading: false })
    }
  }, [report])

  const refreshList = async (preferredReportId?: string) => {
    setLoading(true)
    setError(null)
    try {
      const list = await window.api.analysis.listReports(repoId, 50)
      setReports(list)
      if (list.length === 0) {
        setSelectedReportId('')
        setReport(null)
        return
      }
      const preferred = preferredReportId || reportIdInUrl || selectedReportId
      const stillExists = preferred ? list.some((r) => r.id === preferred) : false
      setSelectedReportId(stillExists ? (preferred as string) : list[0].id)
    } catch (err) {
      setError(toErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  const deleteSelectedReport = async () => {
    if (!report) return
    setDeleting(true)
    setError(null)
    try {
      const deletedId = report.summary.id
      await window.api.analysis.deleteReport(deletedId)
      setConfirmDelete(false)
      // Navigate first so reportIdInUrl is cleared before refreshList updates
      // selectedReportId — otherwise the fetch effect would try to load the
      // just-deleted report (it prefers reportIdInUrl over selectedReportId).
      navigate('/reports')
      await refreshList()
    } catch (err) {
      setError(toErrorMessage(err))
    } finally {
      setDeleting(false)
    }
  }

  const rerunSection = useCallback(async (letter: string) => {
    if (!report) return
    setRerunLetter(letter)
    setError(null)
    try {
      await window.api.analysis.rerunSection({
        report_id: report.summary.id,
        section: letter,
        provider_id: report.summary.provider_id,
        model_id: report.summary.model_id,
      })
      const fresh = await window.api.analysis.getReport(report.summary.id)
      setReport(fresh)
    } catch (err) {
      setError(toErrorMessage(err))
    } finally {
      setRerunLetter(null)
    }
  }, [report])

  const exportAudit = async () => {
    if (!report) return
    setExportAuditBusy(true)
    setError(null)
    try {
      const out = await window.api.analysis.exportAuditSection(report.summary.id)
      if (out.saved && out.file_path) {
        toast.success(`Audit markdown saved: ${out.file_path}`)
      }
    } catch (err) {
      setError(toErrorMessage(err))
    } finally {
      setExportAuditBusy(false)
    }
  }

  const runCompare = async () => {
    if (!report || !compareOtherId.trim()) return
    setCompareBusy(true)
    setError(null)
    try {
      const res = await window.api.analysis.compareReports({
        report_id_a: report.summary.id,
        report_id_b: compareOtherId.trim(),
      })
      setDiffResult(res)
      setCompareOpen(false)
    } catch (err) {
      setError(toErrorMessage(err))
    } finally {
      setCompareBusy(false)
    }
  }

  const exportMarkdown = async () => {
    if (!report) return
    setExportingMd(true)
    setError(null)
    try {
      const out = await window.api.analysis.exportReportMarkdown(report.summary.id)
      if (out.saved && out.file_path) {
        toast.success(`Markdown saved: ${out.file_path}`)
      }
    } catch (err) {
      setError(toErrorMessage(err))
    } finally {
      setExportingMd(false)
    }
  }

  useEffect(() => {
    void refreshList()
  }, [repoId, reportIdInUrl])

  useEffect(() => {
    if (!isDetailMode || !report) {
      setRepoReportsForCompare([])
      return
    }
    const rid = report.summary.repo_id
    void window.api.analysis.listReports(rid, 50).then(setRepoReportsForCompare)
  }, [isDetailMode, report?.summary.repo_id, report?.summary.id])

  useEffect(() => {
    const run = async () => {
      if (!isDetailMode) {
        setReport(null)
        return
      }
      const targetId = reportIdInUrl || selectedReportId
      if (!targetId) {
        setReport(null)
        return
      }
      try {
        const out = await window.api.analysis.getReport(targetId)
        setReport(out)
      } catch (err) {
        const msg = toErrorMessage(err)
        // Report was deleted externally — silently refresh the list instead of showing an error
        if (msg.toLowerCase().includes('not found')) {
          navigate('/reports')
          await refreshList()
        } else {
          setError(msg)
        }
      }
    }
    run()
  }, [isDetailMode, selectedReportId, reportIdInUrl])

  useEffect(() => {
    if (!report) {
      setStaleness(null)
      return
    }
    setStalenessLoading(true)
    window.api.analysis.getStaleness(report.summary.id)
      .then(setStaleness)
      .catch(() => {})
      .finally(() => setStalenessLoading(false))
  }, [report?.summary.id])

  const sectionsV2 = useMemo(() => {
    if (!report?.report) return null
    return getReportSections(report.report as unknown)
  }, [report])

  const diffChangedSections = useMemo(() => {
    if (!diffResult) return []
    return Object.entries(diffResult.section_diffs).filter(([, d]) => d.changed)
  }, [diffResult])

  const v2Ok = (letter: string): boolean => {
    if (!sectionsV2)
      return false
    const block = sectionsV2[letter]
    if (!block || typeof block !== 'object')
      return false
    return !('error' in block)
  }

  return (
    <div className="flex flex-col h-full">
      <div className="screen-header">
        <h1 className="screen-title">Reports</h1>
        <p className="screen-subtitle">View generated analysis artifacts</p>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}
        {!isDetailMode ? (
          <div className="bg-zinc-900/60 border border-zinc-700 rounded-xl p-3 space-y-2">
            <div className="text-xs font-semibold text-zinc-100">Selection Only</div>
            {loading ? (
              <div className="text-xs text-zinc-500">Loading...</div>
            ) : reports.length === 0 ? (
              <div className="text-xs text-zinc-500">No report yet. Run analysis first.</div>
            ) : (
              <>
                <div className="space-y-1.5 max-h-[64vh] overflow-auto">
                  {reports.map((r) => (
                    <button
                      key={r.id}
                      onClick={() => setSelectedReportId(r.id)}
                      onDoubleClick={() => navigate(`/reports?reportId=${encodeURIComponent(r.id)}`)}
                      className={`w-full text-left rounded-md border px-2 py-1.5 text-[11px] ${
                        r.id === selectedReportId
                          ? 'border-indigo-600 bg-indigo-950/40 text-indigo-200'
                          : 'border-zinc-700 bg-zinc-950 text-zinc-300 hover:border-zinc-600'
                      }`}
                    >
                      <div className="font-mono truncate">{r.id.slice(0, 10)}</div>
                      <div className="grid grid-cols-[54px,1fr] gap-x-2 gap-y-0.5 mt-1 text-[11px]">
                        <div className="text-zinc-500">repo</div>
                        <div className="text-zinc-300 truncate">{r.repo_name || r.repo_id}</div>
                        <div className="text-zinc-500">branch</div>
                        <div className="text-zinc-300 truncate">{r.branch || 'unknown'}</div>
                        <div className="text-zinc-500">model</div>
                        <div className="text-zinc-300 truncate">{r.model_id}</div>
                      </div>
                      <div className="text-zinc-500 truncate mt-1">{r.created_at}</div>
                    </button>
                  ))}
                </div>
                <div className="pt-1">
                  <button
                    onClick={() => selectedReportId && navigate(`/reports?reportId=${encodeURIComponent(selectedReportId)}`)}
                    disabled={!selectedReportId}
                    className="px-3 py-1.5 text-xs bg-indigo-600 hover:bg-indigo-500 text-white rounded-md disabled:opacity-50"
                  >
                    Select
                  </button>
                </div>
              </>
            )}
          </div>
        ) : (
          <div className="bg-zinc-900/60 border border-zinc-700 rounded-xl p-3 space-y-3">
            {report ? (
              <>
                <div className="flex items-center justify-between gap-2">
                  <div className="text-xs text-zinc-300">
                    repo: <span className="text-zinc-200 font-mono">{report.summary.repo_name || report.summary.repo_id}</span>
                    <span className="mx-2 text-zinc-700">|</span>
                    branch: <span className="text-zinc-200 font-mono">{report.summary.branch || 'unknown'}</span>
                    <span className="mx-2 text-zinc-700">|</span>
                    model: <span className="text-zinc-200 font-mono">{report.summary.model_id}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => navigate('/reports')}
                      className="px-2 py-1 text-[11px] border border-zinc-700 text-zinc-300 rounded hover:border-zinc-600"
                    >
                      Back
                    </button>
                    {repoReportsForCompare.length >= 2 && (
                      <button
                        type="button"
                        onClick={() => {
                          setCompareOtherId(
                            repoReportsForCompare.find((r) => r.id !== report.summary.id)?.id ?? ''
                          )
                          setCompareOpen(true)
                        }}
                        className="px-2 py-1 text-[11px] border border-sky-800 text-sky-300 rounded hover:border-sky-600"
                      >
                        Compare with…
                      </button>
                    )}
                    <button
                      onClick={() => { void exportMarkdown() }}
                      disabled={exportingMd}
                      className="px-2 py-1 text-[11px] border border-indigo-700 text-indigo-300 rounded hover:border-indigo-500 disabled:opacity-50"
                    >
                      {exportingMd ? 'Exporting...' : 'Export .md'}
                    </button>
                    <button
                      onClick={() => {
                        if (hideDeleteWarning) {
                          void deleteSelectedReport()
                          return
                        }
                        setConfirmDelete(true)
                      }}
                      disabled={deleting}
                      className="px-2 py-1 text-[11px] border border-rose-700 text-rose-300 rounded hover:border-rose-500 disabled:opacity-50"
                    >
                      {deleting ? 'Deleting...' : 'Delete report'}
                    </button>
                  </div>
                </div>
                {diffResult && (
                  <div className="rounded-lg border border-zinc-600/80 bg-zinc-950/60 p-3 space-y-2">
                    <div className="flex flex-wrap items-center gap-2 justify-between">
                      <div className="flex flex-wrap items-center gap-2">
                        <span
                          className={`text-[10px] uppercase tracking-wide px-2 py-0.5 rounded border ${
                            diffResult.quality_trend === 'improving'
                              ? 'border-emerald-800 text-emerald-300 bg-emerald-950/30'
                              : diffResult.quality_trend === 'degrading'
                                ? 'border-rose-800 text-rose-300 bg-rose-950/30'
                                : 'border-zinc-600 text-zinc-400 bg-zinc-900/40'
                          }`}
                        >
                          {diffResult.quality_trend}
                        </span>
                        <span className="text-[11px] text-zinc-300">
                          {diffResult.identical
                            ? 'No section differences'
                            : `${diffResult.sections_changed} section(s) changed`}
                        </span>
                      </div>
                      <button
                        type="button"
                        onClick={() => setDiffResult(null)}
                        className="text-[11px] text-zinc-500 hover:text-zinc-300"
                      >
                        Dismiss
                      </button>
                    </div>
                    <div className="max-h-56 overflow-y-auto space-y-1">
                      {diffChangedSections.map(([letter, d]) => (
                          <div
                            key={letter}
                            className={`text-[11px] rounded border px-2 py-1.5 ${
                              d.improvement === true
                                ? 'border-emerald-800/70 bg-emerald-950/15'
                                : d.improvement === false
                                  ? 'border-rose-800/70 bg-rose-950/15'
                                  : 'border-zinc-700 bg-zinc-900/30'
                            }`}
                          >
                            <span className="font-mono font-semibold text-zinc-200">{letter}</span>
                            {d.confidence_delta && (
                              <span className="text-zinc-300"> · {d.confidence_delta}</span>
                            )}
                            {d.list_added.length > 0 && (
                              <span className="text-emerald-400/90"> · +{d.list_added.length}</span>
                            )}
                            {d.list_removed.length > 0 && (
                              <span className="text-rose-400/90"> · −{d.list_removed.length}</span>
                            )}
                          </div>
                        ))}
                    </div>
                  </div>
                )}
                {staleness && staleness.stale && (staleness.changed_files_count ?? 0) > 0 && (
                  <div className={`rounded-lg border px-3 py-2 text-xs ${
                    staleness.recommend_new_snapshot
                      ? 'border-amber-800/60 bg-amber-950/30 text-amber-200'
                      : 'border-yellow-800/60 bg-yellow-950/30 text-yellow-200'
                  }`}>
                    {staleness.recommend_new_snapshot ? (
                      <div>
                        <strong>{staleness.changed_files_count} files</strong> changed since analysis ({staleness.insertions} insertions, {staleness.deletions} deletions) — recommend rebuilding snapshot before re-running sections
                      </div>
                    ) : (
                      <div>
                        <strong>{staleness.changed_files_count} files</strong> changed since analysis
                        {staleness.sections_affected && staleness.sections_affected.length > 0 && (
                          <span> — sections {staleness.sections_affected.join(', ')} may be stale</span>
                        )}
                      </div>
                    )}
                  </div>
                )}
                {sectionsV2 ? (
                  <div className="space-y-2">
                    {REPORT_SECTION_ORDER.map((letter) => {
                      // Section L: if absent, show a generate prompt instead of nothing
                      if (letter === 'L' && !v2Ok('L')) {
                        return (
                          <div key="L" className="rounded-xl border border-zinc-700/50 bg-zinc-900/40 px-4 py-3 flex items-center justify-between gap-3">
                            <div>
                              <div className="text-sm font-semibold text-zinc-300">Synthesis Report</div>
                              <div className="text-xs text-zinc-500 mt-0.5">Not generated for this run — generates an executive summary across all sections.</div>
                            </div>
                            <Button
                              variant="secondary"
                              size="sm"
                              onClick={() => void rerunSection('L')}
                              loading={rerunLetter === 'L'}
                              disabled={!!rerunLetter}
                            >
                              Generate
                            </Button>
                          </div>
                        )
                      }
                      if (!v2Ok(letter)) return null
                      const Comp = SECTION_COMPONENTS[letter]
                      if (!Comp) return null
                      const extraCtx: ExtraPropsCtx = { exportAudit, exportAuditBusy }
                      const extraProps = SECTION_EXTRA_PROPS[letter]?.(extraCtx) ?? {}
                      // Section A gets enrichment from B (entrypoints) and F (feature names)
                      const sectionAExtras = letter === 'A' ? {
                        featureNames: ((sectionsV2['F'] as SectionF | undefined)?.features ?? [])
                          .map((f) => f.name).filter(Boolean),
                        entrypoints: (sectionsV2['B'] as SectionB | undefined)?.entrypoints ?? [],
                      } : {}
                      const isStale = staleness?.stale && staleness.sections_affected?.includes(letter)
                      return (
                        <ErrorBoundary key={letter} fallback={<div className="rounded-lg border border-red-900/50 bg-red-950/20 px-3 py-2 text-xs text-red-400">Section {letter} failed to render</div>}>
                          <Comp
                            data={sectionsV2[letter] as never}
                            onRerun={() => void rerunSection(letter)}
                            rerunBusy={rerunLetter === letter}
                            onShowSources={() => showSources(letter)}
                            isStale={isStale}
                            {...extraProps}
                            {...sectionAExtras}
                          />
                        </ErrorBoundary>
                      )
                    })}
                  </div>
                ) : (
                  <div className="text-xs text-zinc-500 py-4 text-center">
                    No analysis sections found. Re-run analysis to generate section data.
                  </div>
                )}
              </>
            ) : (
              <div className="text-xs text-zinc-500">Loading report...</div>
            )}
          </div>
        )}
      </div>
      <Modal open={compareOpen && !!report} onClose={() => setCompareOpen(false)}>
        <Modal.Overlay />
        <Modal.Panel className="w-full max-w-md">
          <Modal.Header>Compare reports</Modal.Header>
          <div className="px-4 py-3 space-y-3 border-b border-zinc-700">
            <div className="text-xs text-zinc-300">
              Select another run from the same repository to diff against this report.
            </div>
            <select
              value={compareOtherId}
              onChange={(e) => setCompareOtherId(e.target.value)}
              className="w-full rounded border border-zinc-700 bg-zinc-950 text-xs text-zinc-200 px-2 py-2"
            >
              {repoReportsForCompare
                .filter((r) => r.id !== report?.summary.id)
                .map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.id.slice(0, 10)} — {r.model_id} — {r.created_at}
                  </option>
                ))}
            </select>
          </div>
          <Modal.Footer>
            <Button variant="secondary" onClick={() => setCompareOpen(false)} disabled={compareBusy}>
              Cancel
            </Button>
            <Button variant="primary" onClick={() => void runCompare()} disabled={compareBusy || !compareOtherId} loading={compareBusy}>
              Compare
            </Button>
          </Modal.Footer>
        </Modal.Panel>
      </Modal>
      <ConfirmDialog
        open={confirmDelete && !!report}
        onClose={() => setConfirmDelete(false)}
        onConfirm={() => void deleteSelectedReport()}
        title="Delete this analysis report?"
        description={
          <>
            <p className="mb-3">This removes the generated report artifact from local database.</p>
            {!hideDeleteWarning && (
              <label className="inline-flex items-center gap-2 text-xs text-zinc-300">
                <input
                  type="checkbox"
                  onChange={(e) => {
                    const v = e.target.checked
                    setHideDeleteWarning(v)
                    localStorage.setItem('reports.deleteWarningHidden', v ? '1' : '0')
                  }}
                />
                Do not show this again
              </label>
            )}
          </>
        }
        confirmLabel="Delete"
        confirmVariant="danger"
        loading={deleting}
      />
      {sourcesPanel && (
        <>
          <div
            className="fixed inset-0 bg-black/40 z-40"
            onClick={() => setSourcesPanel(null)}
          />
          <EvidencePanel
            sectionId={sourcesPanel.sectionId}
            sources={sourcesPanel.sources}
            loading={sourcesPanel.loading}
            onClose={() => setSourcesPanel(null)}
          />
        </>
      )}
    </div>
  )
}
