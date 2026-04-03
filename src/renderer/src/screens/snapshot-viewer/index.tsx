import React, { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { ArrowLeft, ChevronDown, ChevronRight, FileText, Folder, Loader2 } from 'lucide-react'
import type { ManifestTreeNode } from '../../types/electron'
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
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const [ignoredPaths, setIgnoredPaths] = useState<Set<string>>(new Set())
  const [completing, setCompleting] = useState(false)
  const [completeDone, setCompleteDone] = useState(false)
  const [activating, setActivating] = useState(false)

  const tree = useMemo(() => buildTree(treeNodes), [treeNodes])

  useEffect(() => {
    const run = async () => {
      if (!snapshotId) return
      setLoadingTree(true)
      setError(null)
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
          onClick={() => setSelectedFilePath(node.path)}
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
            <button
              onClick={async () => {
                if (!snapshotId) return
                setCompleting(true)
                setError(null)
                try {
                  await window.api.manifest.build(snapshotId, Array.from(ignoredPaths))
                  const res = await window.api.manifest.tree(snapshotId)
                  setTreeNodes(res.nodes)
                  if (selectedFilePath && Array.from(ignoredPaths).some((p) => p === selectedFilePath || (p.endsWith('/**') && selectedFilePath.startsWith(p.slice(0, -3))))) {
                    setSelectedFilePath(res.nodes.find((n) => !n.is_dir)?.path ?? null)
                  }
                  setCompleteDone(true)
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
                if (!repoId || !snapshotId) return
                setActivating(true)
                setError(null)
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

        <div className="grid grid-cols-12 gap-3 h-[calc(100%-3.25rem)]">
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
                      <div key={`ln-${i + 1}`}>{i + 1}</div>
                    ))}
                  </pre>
                  <pre className="text-zinc-200 whitespace-pre overflow-x-auto">
                    {lines.map((line, i) => (
                      <div key={`lc-${i + 1}`}>{line || ' '}</div>
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
