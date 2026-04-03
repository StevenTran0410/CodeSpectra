import React, { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { ArrowLeft, Loader2, Save, Search } from 'lucide-react'
import type { RepoMapSummary, SymbolRecord } from '../../types/electron'
import { ErrorBanner } from '../../components/ui/ErrorBanner'

export default function IndexOverviewScreen(): React.ReactElement {
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const repoId = params.get('repoId') ?? ''
  const snapshotId = params.get('snapshotId') ?? ''

  const [loading, setLoading] = useState(false)
  const [savingCsv, setSavingCsv] = useState(false)
  const [excludeTestsOnExport, setExcludeTestsOnExport] = useState(true)
  const [summary, setSummary] = useState<RepoMapSummary | null>(null)
  const [symbols, setSymbols] = useState<SymbolRecord[]>([])
  const [query, setQuery] = useState('')
  const [hits, setHits] = useState<SymbolRecord[]>([])
  const [searching, setSearching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  useEffect(() => {
    const run = async () => {
      if (!snapshotId) return
      setLoading(true)
      setError(null)
      setSuccess(null)
      try {
        const s = await window.api.repomap.summary(snapshotId)
        setSummary(s)
        const sample = await window.api.repomap.symbols(snapshotId, 200)
        setSymbols(sample.symbols)
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
      } finally {
        setLoading(false)
      }
    }
    run()
  }, [snapshotId])

  useEffect(() => {
    const run = async () => {
      if (!snapshotId || !query.trim()) {
        setHits([])
        return
      }
      setSearching(true)
      try {
        const res = await window.api.repomap.search(snapshotId, query.trim(), 120)
        setHits(res.symbols)
      } finally {
        setSearching(false)
      }
    }
    const timer = setTimeout(run, 180)
    return () => clearTimeout(timer)
  }, [snapshotId, query])

  const byKind = useMemo(() => Object.entries(summary?.kind_breakdown ?? {}), [summary])
  const byLang = useMemo(() => Object.entries(summary?.language_breakdown ?? {}), [summary])

  return (
    <>
      <div className="screen-header">
        <h1 className="screen-title">Index Overview</h1>
        <p className="screen-subtitle">Deep index quality, symbol breakdown, and export</p>
      </div>
      <div className="h-[calc(100vh-10rem)] p-4 space-y-3 overflow-y-auto">
        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}
        {success && (
          <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 text-emerald-300 text-xs px-3 py-2">
            {success}
          </div>
        )}

        <div className="flex items-center justify-between bg-zinc-900/60 border border-zinc-700 rounded-lg px-3 py-2">
          <div className="text-xs text-zinc-400">
            <span className="text-zinc-200">Repo:</span> <span className="font-mono">{repoId || '-'}</span>
            <span className="mx-2 text-zinc-600">|</span>
            <span className="text-zinc-200">Snapshot:</span> <span className="font-mono">{snapshotId || '-'}</span>
          </div>
          <div className="flex items-center gap-2">
            <label className="inline-flex items-center gap-1.5 text-[11px] text-zinc-400">
              <input
                type="checkbox"
                checked={excludeTestsOnExport}
                onChange={(e) => setExcludeTestsOnExport(e.target.checked)}
              />
              Exclude tests in CSV
            </label>
            <button
              onClick={async () => {
                if (!snapshotId) return
                setSavingCsv(true)
                setError(null)
                setSuccess(null)
                try {
                  const out = await window.api.repomap.exportCsv(snapshotId, excludeTestsOnExport)
                  if (out.saved && out.file_path) {
                    setSuccess(`CSV saved: ${out.file_path} (${out.row_count} rows)`)
                  }
                } catch (err) {
                  setError(err instanceof Error ? err.message : String(err))
                } finally {
                  setSavingCsv(false)
                }
              }}
              disabled={savingCsv || !summary}
              className="px-2.5 py-1.5 text-xs border border-zinc-700 rounded-md text-zinc-300 hover:border-zinc-600 disabled:opacity-50 inline-flex items-center gap-1"
            >
              {savingCsv ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />}
              Save index CSV
            </button>
            <button
              onClick={() => navigate(`/snapshot-viewer?repoId=${encodeURIComponent(repoId)}&snapshotId=${encodeURIComponent(snapshotId)}`)}
              className="px-2.5 py-1.5 text-xs border border-zinc-700 rounded-md text-zinc-300 hover:border-zinc-600 inline-flex items-center gap-1"
            >
              <ArrowLeft size={12} />
              Back
            </button>
          </div>
        </div>

        {loading ? (
          <div className="text-xs text-zinc-500 inline-flex items-center gap-2">
            <Loader2 size={12} className="animate-spin" />
            Loading index summary...
          </div>
        ) : summary ? (
          <>
            <div className="grid grid-cols-4 gap-2">
              <div className="bg-zinc-900/60 border border-zinc-700 rounded-md p-3">
                <div className="text-[11px] text-zinc-500">Symbols</div>
                <div className="text-lg font-semibold text-zinc-100">{summary.total_symbols}</div>
              </div>
              <div className="bg-zinc-900/60 border border-zinc-700 rounded-md p-3">
                <div className="text-[11px] text-zinc-500">Files indexed</div>
                <div className="text-lg font-semibold text-zinc-100">{summary.files_indexed}</div>
              </div>
              <div className="bg-zinc-900/60 border border-zinc-700 rounded-md p-3">
                <div className="text-[11px] text-zinc-500">Parse failures</div>
                <div className="text-lg font-semibold text-zinc-100">{summary.parse_failures}</div>
              </div>
              <div className="bg-zinc-900/60 border border-zinc-700 rounded-md p-3">
                <div className="text-[11px] text-zinc-500">Mode</div>
                <div className="text-lg font-semibold text-zinc-100">{summary.extract_mode}</div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="bg-zinc-900/60 border border-zinc-700 rounded-md p-3 space-y-2">
                <div className="text-xs font-semibold text-zinc-100">By symbol kind</div>
                {byKind.map(([k, v]) => (
                  <div key={k} className="flex justify-between text-xs text-zinc-400">
                    <span>{k}</span>
                    <span className="text-zinc-200">{v}</span>
                  </div>
                ))}
              </div>
              <div className="bg-zinc-900/60 border border-zinc-700 rounded-md p-3 space-y-2">
                <div className="text-xs font-semibold text-zinc-100">By language</div>
                {byLang.map(([k, v]) => (
                  <div key={k} className="flex justify-between text-xs text-zinc-400">
                    <span>{k}</span>
                    <span className="text-zinc-200">{v}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-zinc-900/60 border border-zinc-700 rounded-lg p-3 space-y-2">
              <div className="text-xs font-semibold text-zinc-100">Symbol Search</div>
              <div className="flex items-center gap-2">
                <div className="flex-1 relative">
                  <Search size={12} className="absolute left-2 top-2.5 text-zinc-500" />
                  <input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Search symbol name or file path..."
                    className="w-full bg-zinc-950 border border-zinc-700 rounded-md pl-7 pr-2 py-1.5 text-xs text-zinc-200"
                  />
                </div>
                {searching && <Loader2 size={13} className="animate-spin text-zinc-500" />}
              </div>
              {(hits.length > 0 ? hits : symbols).slice(0, 120).map((s) => (
                <div key={s.id} className="text-[11px] text-zinc-400 font-mono border-b border-zinc-800 py-1">
                  <span className="text-zinc-200">{s.name}</span>
                  <span className="mx-2 text-zinc-600">·</span>
                  <span>{s.kind}/{s.extract_source}</span>
                  <span className="mx-2 text-zinc-600">·</span>
                  <span>{s.rel_path}:{s.line_start}</span>
                </div>
              ))}
            </div>
          </>
        ) : null}
      </div>
    </>
  )
}
