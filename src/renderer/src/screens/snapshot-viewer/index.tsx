import React, { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { ArrowLeft, ChevronDown, ChevronRight, FileText, Folder, Loader2, Search } from 'lucide-react'
import type { ManifestTreeNode, RepoMapSummary, SymbolRecord } from '../../types/electron'
import { ErrorBanner } from '../../components/ui/ErrorBanner'

type TreeNode = {
  name: string
  path: string
  isDir: boolean
  children: TreeNode[]
}

function buildTree(nodes: ManifestTreeNode[]): TreeNode[] {
  const root: TreeNode = { name: '', path: '', isDir: true, children: [] }
  const map = new Map<string, TreeNode>()
  map.set('', root)

  const sorted = [...nodes].sort((a, b) => a.path.localeCompare(b.path))
  for (const n of sorted) {
    const parts = n.path.split('/').filter(Boolean)
    let current = root
    let currentPath = ''
    for (let i = 0; i < parts.length; i += 1) {
      const part = parts[i]
      currentPath = currentPath ? `${currentPath}/${part}` : part
      const isLeaf = i === parts.length - 1
      const isDir = isLeaf ? n.is_dir : true
      let next = map.get(currentPath)
      if (!next) {
        next = { name: part, path: currentPath, isDir, children: [] }
        current.children.push(next)
        map.set(currentPath, next)
      }
      current = next
    }
  }

  const sortRec = (arr: TreeNode[]) => {
    arr.sort((a, b) => {
      if (a.isDir !== b.isDir) return a.isDir ? -1 : 1
      return a.name.localeCompare(b.name)
    })
    for (const item of arr) sortRec(item.children)
  }
  sortRec(root.children)
  return root.children
}

export default function SnapshotViewerScreen(): React.ReactElement {
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const repoId = params.get('repoId') ?? ''
  const snapshotId = params.get('snapshotId') ?? ''

  const [loadingTree, setLoadingTree] = useState(false)
  const [loadingFile, setLoadingFile] = useState(false)
  const [treeNodes, setTreeNodes] = useState<ManifestTreeNode[]>([])
  const [selectedFilePath, setSelectedFilePath] = useState<string | null>(null)
  const [fileContent, setFileContent] = useState('')
  const [fileTruncated, setFileTruncated] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const [ignoredPaths, setIgnoredPaths] = useState<Set<string>>(new Set())
  const [completing, setCompleting] = useState(false)
  const [completeDone, setCompleteDone] = useState(false)
  const [activating, setActivating] = useState(false)
  const [indexing, setIndexing] = useState(false)
  const [repoMapSummary, setRepoMapSummary] = useState<RepoMapSummary | null>(null)
  const [previewSymbols, setPreviewSymbols] = useState<SymbolRecord[]>([])
  const [symbolQuery, setSymbolQuery] = useState('')
  const [symbolResults, setSymbolResults] = useState<SymbolRecord[]>([])
  const [searchingSymbols, setSearchingSymbols] = useState(false)
  const [focusedLine, setFocusedLine] = useState<number | null>(null)
  const [exportingCsv, setExportingCsv] = useState(false)
  const [excludeTestsOnExport, setExcludeTestsOnExport] = useState(true)
  const runtimeHasExportCsv = typeof window.api?.repomap?.exportCsv === 'function'

  const tree = useMemo(() => buildTree(treeNodes), [treeNodes])

  useEffect(() => {
    const run = async () => {
      if (!snapshotId) return
      setLoadingTree(true)
      setError(null)
      setSuccess(null)
      try {
        const snap = await window.api.sync.getSnapshot(snapshotId)
        const restoredIgnores = snap.manual_ignores ?? []
        setIgnoredPaths(new Set(restoredIgnores))
        setCompleteDone(true)
        await window.api.manifest.build(snapshotId, restoredIgnores)
        const res = await window.api.manifest.tree(snapshotId)
        setTreeNodes(res.nodes)
        const firstFile = res.nodes.find((n) => !n.is_dir)?.path ?? null
        setSelectedFilePath(firstFile)

        const defaults = new Set<string>()
        for (const n of res.nodes) {
          if (n.is_dir) defaults.add(n.path)
        }
        setExpanded(defaults)
        try {
          const summary = await window.api.repomap.summary(snapshotId)
          setRepoMapSummary(summary)
          const sample = await window.api.repomap.symbols(snapshotId, 40)
          setPreviewSymbols(sample.symbols)
        } catch {
          setRepoMapSummary(null)
          setPreviewSymbols([])
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
      } finally {
        setLoadingTree(false)
      }
    }
    run()
  }, [snapshotId])

  useEffect(() => {
    const run = async () => {
      if (!snapshotId || !selectedFilePath) {
        setFileContent('')
        setFileTruncated(false)
        return
      }
      setLoadingFile(true)
      setError(null)
      setSuccess(null)
      try {
        const res = await window.api.manifest.file(snapshotId, selectedFilePath)
        setFileContent(res.content)
        setFileTruncated(res.truncated)
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
      } finally {
        setLoadingFile(false)
      }
    }
    run()
  }, [snapshotId, selectedFilePath])

  useEffect(() => {
    const run = async () => {
      if (!snapshotId || !symbolQuery.trim()) {
        setSymbolResults([])
        return
      }
      setSearchingSymbols(true)
      try {
        const res = await window.api.repomap.search(snapshotId, symbolQuery.trim(), 80)
        setSymbolResults(res.symbols)
      } finally {
        setSearchingSymbols(false)
      }
    }
    const timer = setTimeout(run, 180)
    return () => clearTimeout(timer)
  }, [snapshotId, symbolQuery])

  const renderNode = (node: TreeNode, depth: number): React.ReactElement => {
    const isOpen = expanded.has(node.path)
    if (node.isDir) {
      return (
        <div key={node.path}>
          <div className="w-full flex items-center gap-1 text-xs px-2 py-1 text-zinc-300 hover:bg-zinc-800/80">
            <button
              onClick={() => {
                setExpanded((prev) => {
                  const next = new Set(prev)
                  if (next.has(node.path)) next.delete(node.path)
                  else next.add(node.path)
                  return next
                })
              }}
              className="inline-flex items-center gap-1 flex-1"
              style={{ paddingLeft: `${8 + depth * 14}px` }}
            >
              {isOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              <Folder size={12} className="text-zinc-400" />
              <span className="truncate">{node.name}</span>
            </button>
            <button
              onClick={() => {
                setIgnoredPaths((prev) => {
                  const next = new Set(prev)
                  const key = `${node.path}/**`
                  if (next.has(key)) next.delete(key)
                  else next.add(key)
                  return next
                })
                setCompleteDone(false)
              }}
              className={`px-1.5 py-0.5 rounded text-[10px] border ${
                ignoredPaths.has(`${node.path}/**`)
                  ? 'border-amber-400/50 text-amber-300'
                  : 'border-zinc-700 text-zinc-500'
              }`}
            >
              Ignore
            </button>
          </div>
          {isOpen && node.children.map((child) => renderNode(child, depth + 1))}
        </div>
      )
    }

    return (
      <div
        key={node.path}
        className={`w-full flex items-center gap-1 text-xs px-2 py-1 hover:bg-zinc-800/80 ${
          selectedFilePath === node.path ? 'bg-blue-500/15 text-blue-200' : 'text-zinc-300'
        }`}
      >
        <button
          onClick={() => {
            setSelectedFilePath(node.path)
            setFocusedLine(null)
          }}
          className="inline-flex items-center gap-1 flex-1"
          style={{ paddingLeft: `${22 + depth * 14}px` }}
          title={node.path}
        >
          <FileText size={12} className="text-zinc-500" />
          <span className="truncate">{node.name}</span>
        </button>
        <button
          onClick={() => {
            setIgnoredPaths((prev) => {
              const next = new Set(prev)
              if (next.has(node.path)) next.delete(node.path)
              else next.add(node.path)
              return next
            })
            setCompleteDone(false)
          }}
          className={`px-1.5 py-0.5 rounded text-[10px] border ${
            ignoredPaths.has(node.path)
              ? 'border-amber-400/50 text-amber-300'
              : 'border-zinc-700 text-zinc-500'
          }`}
        >
          Ignore
        </button>
      </div>
    )
  }

  const lines = fileContent.split('\n')

  return (
    <>
      <div className="screen-header">
        <h1 className="screen-title">Snapshot Viewer</h1>
        <p className="screen-subtitle">Read-only source tree and code preview</p>
      </div>
      <div className="h-[calc(100vh-10rem)] p-4 space-y-3">
        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}
        {success && (
          <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 text-emerald-300 text-xs px-3 py-2">
            {success}
          </div>
        )}

        <div className="flex items-center justify-between bg-zinc-900/60 border border-zinc-700 rounded-lg px-3 py-2">
          <div className="text-xs text-zinc-400 flex items-center gap-2">
            <span className="text-zinc-200">Repo:</span> <span className="font-mono">{repoId || '-'}</span>
            <span className="mx-2 text-zinc-600">|</span>
            <span className="text-zinc-200">Snapshot:</span> <span className="font-mono">{snapshotId || '-'}</span>
            <span className="mx-2 text-zinc-600">|</span>
            <span>Ignore selected: <span className="text-zinc-200">{ignoredPaths.size}</span></span>
            {completeDone && <span className="text-emerald-400">Completed</span>}
          </div>
          <div className="flex items-center gap-2">
            <label className="inline-flex items-center gap-1.5 text-[11px] text-zinc-400 mr-1">
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
                setCompleting(true)
                setError(null)
                setSuccess(null)
                try {
                  await window.api.manifest.build(snapshotId, Array.from(ignoredPaths))
                  const res = await window.api.manifest.tree(snapshotId)
                  setTreeNodes(res.nodes)
                  if (selectedFilePath && Array.from(ignoredPaths).some((p) => p === selectedFilePath || (p.endsWith('/**') && selectedFilePath.startsWith(p.slice(0, -3))))) {
                    setSelectedFilePath(res.nodes.find((n) => !n.is_dir)?.path ?? null)
                  }
                  setCompleteDone(true)
                  setRepoMapSummary(null)
                  setPreviewSymbols([])
                } catch (err) {
                  setError(err instanceof Error ? err.message : String(err))
                } finally {
                  setCompleting(false)
                }
              }}
              disabled={completing}
              className="px-2.5 py-1.5 text-xs border border-zinc-700 rounded-md text-zinc-300 hover:border-zinc-600 disabled:opacity-50"
            >
              {completing ? 'Completing...' : 'Complete'}
            </button>
            <button
              onClick={async () => {
                if (!snapshotId) return
                setIndexing(true)
                setError(null)
                setSuccess(null)
                try {
                  const built = await window.api.repomap.build(snapshotId, true)
                  setRepoMapSummary(built.summary)
                  const sample = await window.api.repomap.symbols(snapshotId, 40)
                  setPreviewSymbols(sample.symbols)
                } catch (err) {
                  setError(err instanceof Error ? err.message : String(err))
                } finally {
                  setIndexing(false)
                }
              }}
              disabled={indexing}
              className="px-2.5 py-1.5 text-xs border border-zinc-700 rounded-md text-zinc-300 hover:border-zinc-600 disabled:opacity-50"
            >
              {indexing ? 'Indexing...' : 'Build deep index'}
            </button>
            <button
              onClick={async () => {
                if (!snapshotId) return
                if (!window.api?.repomap?.exportCsv) {
                  setError('repomap.exportCsv bridge is unavailable. Restart the app (or restart npm run dev).')
                  return
                }
                setExportingCsv(true)
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
                  setExportingCsv(false)
                }
              }}
              disabled={exportingCsv || !repoMapSummary || !runtimeHasExportCsv}
              className="px-2.5 py-1.5 text-xs border border-zinc-700 rounded-md text-zinc-300 hover:border-zinc-600 disabled:opacity-50"
            >
              {exportingCsv ? 'Saving CSV...' : 'Save index CSV'}
            </button>
            <button
              onClick={async () => {
                if (!repoId || !snapshotId) return
                setActivating(true)
                setError(null)
                setSuccess(null)
                try {
                  await window.api.folder.setActiveSnapshot(repoId, snapshotId)
                  navigate(`/repositories?repoId=${encodeURIComponent(repoId)}`)
                } catch (err) {
                  setError(err instanceof Error ? err.message : String(err))
                } finally {
                  setActivating(false)
                }
              }}
              disabled={!completeDone || activating}
              className="px-2.5 py-1.5 text-xs bg-indigo-600 hover:bg-indigo-500 rounded-md text-white disabled:opacity-50"
            >
              {activating ? 'Selecting...' : 'Select for index'}
            </button>
            <button
              onClick={() => navigate('/repositories')}
              className="px-2.5 py-1.5 text-xs border border-zinc-700 rounded-md text-zinc-300 hover:border-zinc-600 inline-flex items-center gap-1"
            >
              <ArrowLeft size={12} />
              Back
            </button>
          </div>
        </div>

        {repoMapSummary && (
          <div className="bg-zinc-900/60 border border-zinc-700 rounded-lg p-3 text-xs text-zinc-300 space-y-2">
            <div className="font-semibold text-zinc-100">Deep Index Summary</div>
            <div className="flex items-center gap-3 text-zinc-400">
              <span>Symbols: <span className="text-zinc-200">{repoMapSummary.total_symbols}</span></span>
              <span>Files: <span className="text-zinc-200">{repoMapSummary.files_indexed}</span></span>
              <span>Parse failures: <span className="text-zinc-200">{repoMapSummary.parse_failures}</span></span>
              <span>Mode: <span className="text-zinc-200">{repoMapSummary.extract_mode}</span></span>
            </div>
            {previewSymbols.length > 0 && (
              <div className="max-h-28 overflow-auto border border-zinc-700 rounded-md">
                {previewSymbols.map((s) => (
                  <div key={s.id} className="px-2 py-1 border-b border-zinc-800 last:border-b-0 font-mono text-[11px] text-zinc-400">
                    <span className="text-zinc-200">{s.name}</span>
                    <span className="mx-2 text-zinc-600">·</span>
                    <span>{s.kind}</span>
                    <span className="mx-2 text-zinc-600">·</span>
                    <span>{s.extract_source}</span>
                    <span className="mx-2 text-zinc-600">·</span>
                    <span>{s.rel_path}:{s.line_start}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <div className="bg-zinc-900/60 border border-zinc-700 rounded-lg p-3 space-y-2">
          <div className="text-xs font-semibold text-zinc-100">Symbol Search</div>
          <div className="flex items-center gap-2">
            <div className="flex-1 relative">
              <Search size={12} className="absolute left-2 top-2.5 text-zinc-500" />
              <input
                value={symbolQuery}
                onChange={(e) => setSymbolQuery(e.target.value)}
                placeholder="Search symbol name or file path..."
                className="w-full bg-zinc-950 border border-zinc-700 rounded-md pl-7 pr-2 py-1.5 text-xs text-zinc-200"
              />
            </div>
            {searchingSymbols && <Loader2 size={13} className="animate-spin text-zinc-500" />}
          </div>
          {symbolResults.length > 0 && (
            <div className="max-h-28 overflow-auto border border-zinc-700 rounded-md">
              {symbolResults.map((s) => (
                <button
                  key={`hit-${s.id}`}
                  onClick={() => {
                    setSelectedFilePath(s.rel_path)
                    setFocusedLine(s.line_start)
                  }}
                  className="w-full text-left px-2 py-1 border-b border-zinc-800 last:border-b-0 hover:bg-zinc-800/50"
                >
                  <div className="text-[11px] text-zinc-200 font-mono">
                    {s.name} <span className="text-zinc-500">({s.kind}/{s.extract_source})</span>
                  </div>
                  <div className="text-[10px] text-zinc-500 font-mono">{s.rel_path}:{s.line_start}</div>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="grid grid-cols-12 gap-3 h-[calc(100%-8.75rem)]">
          <div className="col-span-4 border border-zinc-700 rounded-md overflow-auto bg-zinc-900/40">
            <div className="sticky top-0 z-10 px-2 py-1.5 text-[11px] uppercase tracking-wide text-zinc-500 bg-zinc-900/95 border-b border-zinc-700">
              Explorer
            </div>
            {loadingTree ? (
              <div className="p-3 text-xs text-zinc-500 inline-flex items-center gap-2">
                <Loader2 size={12} className="animate-spin" />
                Loading tree...
              </div>
            ) : tree.length === 0 ? (
              <div className="p-3 text-xs text-zinc-500">No indexed files.</div>
            ) : (
              <div className="py-1">{tree.map((n) => renderNode(n, 0))}</div>
            )}
          </div>

          <div className="col-span-8 border border-zinc-700 rounded-md overflow-auto bg-zinc-950/50">
            <div className="sticky top-0 z-10 px-3 py-1.5 text-[11px] text-zinc-500 bg-zinc-900/95 border-b border-zinc-700 font-mono truncate">
              {selectedFilePath ?? 'Select a file'}
            </div>
            {loadingFile ? (
              <div className="p-3 text-xs text-zinc-500 inline-flex items-center gap-2">
                <Loader2 size={12} className="animate-spin" />
                Loading file...
              </div>
            ) : selectedFilePath ? (
              <div className="p-3 font-mono text-xs">
                {fileTruncated && (
                  <div className="mb-2 text-[11px] text-amber-300">Preview truncated for performance.</div>
                )}
                <div className="grid grid-cols-[56px_1fr] gap-3">
                  <pre className="text-right text-zinc-600 select-none">
                    {lines.map((_line, i) => (
                      <div
                        key={`ln-${i + 1}`}
                        className={focusedLine === i + 1 ? 'bg-amber-500/15 text-amber-300' : ''}
                      >
                        {i + 1}
                      </div>
                    ))}
                  </pre>
                  <pre className="text-zinc-200 whitespace-pre overflow-x-auto">
                    {lines.map((line, i) => (
                      <div
                        key={`lc-${i + 1}`}
                        className={focusedLine === i + 1 ? 'bg-amber-500/15' : ''}
                      >
                        {line || ' '}
                      </div>
                    ))}
                  </pre>
                </div>
              </div>
            ) : (
              <div className="p-3 text-xs text-zinc-500">Select a file from explorer.</div>
            )}
          </div>
        </div>
      </div>
    </>
  )
}
