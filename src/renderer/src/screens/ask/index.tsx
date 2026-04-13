import React, { useState, useRef, useEffect } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { Send, Loader2, X } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { ErrorBanner } from '../../components/ui/ErrorBanner'
import { toErrorMessage } from '../../lib/errors'
import { useLocalRepoStore } from '../../store/local-repo.store'
import { useWorkspaceStore } from '../../store/workspace.store'
import { useQAStore } from '../../store/qa.store'
import type { RepoSnapshot } from '../../types/electron'

export default function AskScreen(): React.ReactElement {
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const urlSnapshotId = params.get('snapshotId') ?? ''
  const urlRepoId = params.get('repoId') ?? ''

  const activeWorkspaceId = useWorkspaceStore((s) => s.activeWorkspaceId)
  const { repos, load: loadRepos } = useLocalRepoStore()

  const {
    conversations,
    activeId,
    createConversation,
    deleteConversation,
    renameConversation,
    updateConversationSnapshot,
    addMessage,
    setActive,
  } = useQAStore()

  const [selectedRepoId, setSelectedRepoId] = useState(urlRepoId)
  const [selectedSnapshotId, setSelectedSnapshotId] = useState(urlSnapshotId)
  const [snapshots, setSnapshots] = useState<RepoSnapshot[]>([])
  const [loadingSnapshots, setLoadingSnapshots] = useState(false)
  const [renamingId, setRenamingId] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const [showChangeSelector, setShowChangeSelector] = useState(false)

  const activeConvo = activeId ? conversations.find((c) => c.id === activeId) : null
  const messages = activeConvo?.messages ?? []

  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  // Load repos on mount
  useEffect(() => {
    loadRepos(activeWorkspaceId ?? undefined)
  }, [loadRepos, activeWorkspaceId])

  // Sync selectors from active conversation on mount or when activeId changes
  useEffect(() => {
    if (activeConvo) {
      setSelectedRepoId(activeConvo.repoId)
      setSelectedSnapshotId(activeConvo.snapshotId)
    }
  }, [activeConvo])

  // Load snapshots when selectedRepoId changes
  useEffect(() => {
    if (!selectedRepoId) {
      setSnapshots([])
      setSelectedSnapshotId('')
      return
    }

    const loadSnapshotsForRepo = async () => {
      try {
        setLoadingSnapshots(true)
        const rows = await window.api.sync.listForRepo(selectedRepoId)
        setSnapshots(rows)
        const ready = rows.find((s) => s.status === 'ready')
        setSelectedSnapshotId(ready?.id || rows[0]?.id || '')
      } catch (err) {
        setError(toErrorMessage(err))
      } finally {
        setLoadingSnapshots(false)
      }
    }

    loadSnapshotsForRepo()
  }, [selectedRepoId])

  // Auto-select first repo if needed
  useEffect(() => {
    if (repos.length === 0) {
      setSelectedRepoId('')
      setSelectedSnapshotId('')
      return
    }
    if (!selectedRepoId || !repos.some((r) => r.id === selectedRepoId)) {
      setSelectedRepoId(repos[0].id)
    }
  }, [repos, selectedRepoId])

  // Get provider/model from localStorage
  const getConfig = () => {
    try {
      const config = JSON.parse(localStorage.getItem('analysis.runConfig.v1') ?? '{}') as {
        providerId?: string
        modelId?: string
      }
      return {
        providerId: config.providerId ?? '',
        modelId: config.modelId ?? '',
      }
    } catch {
      return { providerId: '', modelId: '' }
    }
  }

  const { providerId, modelId } = getConfig()

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
    }
  }, [messages])

  const handleConfirmChange = () => {
    if (!activeId || !selectedSnapshotId || !selectedRepoId) return
    updateConversationSnapshot(activeId, selectedSnapshotId, selectedRepoId)
    setShowChangeSelector(false)
  }

  const handleCreateConversation = () => {
    if (!selectedRepoId || !selectedSnapshotId) {
      setError('Select a repo & snapshot first')
      return
    }
    const id = createConversation(selectedSnapshotId, selectedRepoId)
    setActive(id)
  }

  const handleAsk = async () => {
    const q = input.trim()
    if (!q || loading || !activeId || !activeConvo) return

    setInput('')
    addMessage(activeId, { role: 'user', content: q })
    setLoading(true)
    setError(null)

    try {
      if (!providerId || !modelId) {
        throw new Error('No provider/model configured. Please set up a provider in the Providers screen.')
      }

      const res = await window.api.qa.ask({
        snapshot_id: activeConvo.snapshotId,
        question: q,
        provider_id: providerId,
        model_id: modelId,
      })

      addMessage(activeId, { role: 'assistant', content: res.answer, response: res })
    } catch (err) {
      const msg = toErrorMessage(err)
      addMessage(activeId, { role: 'assistant', content: '', error: msg })
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleAsk()
    }
  }

  const handleTabDoubleClick = (id: string, name: string) => {
    setRenamingId(id)
    setRenameValue(name)
  }

  const handleRenameBlur = (id: string) => {
    if (renameValue.trim()) {
      renameConversation(id, renameValue.trim())
    }
    setRenamingId(null)
    setRenameValue('')
  }

  const handleRenameKeyDown = (e: React.KeyboardEvent<HTMLInputElement>, id: string) => {
    if (e.key === 'Enter') {
      handleRenameBlur(id)
    } else if (e.key === 'Escape') {
      setRenamingId(null)
      setRenameValue('')
    }
  }

  const handleCitationClick = (file: string, lineStart?: number | null) => {
    if (!activeConvo) return
    const url = `/snapshot-viewer?repoId=${encodeURIComponent(activeConvo.repoId)}&snapshotId=${encodeURIComponent(activeConvo.snapshotId)}&file=${encodeURIComponent(file)}${lineStart ? `&line=${lineStart}` : ''}`
    navigate(url)
  }

  return (
    <div className="flex-1 flex flex-col bg-app-bg">
      {/* Tab bar */}
      {conversations.length > 0 && (
        <div className="px-6 py-3 border-b border-surface-border bg-surface-overlay flex items-center gap-2 overflow-x-auto scrollbar-thin">
          {conversations.map((convo) => (
            <div
              key={convo.id}
              className={`flex items-center gap-1 px-3 py-2 rounded-t-lg whitespace-nowrap transition-colors cursor-pointer ${
                activeId === convo.id
                  ? 'bg-surface text-gray-100 border-b-2 border-blue-500'
                  : 'bg-surface-overlay text-gray-400 hover:text-gray-200'
              }`}
              onClick={() => setActive(convo.id)}
            >
              {renamingId === convo.id ? (
                <input
                  type="text"
                  value={renameValue}
                  onChange={(e) => setRenameValue(e.target.value)}
                  onBlur={() => handleRenameBlur(convo.id)}
                  onKeyDown={(e) => handleRenameKeyDown(e, convo.id)}
                  onClick={(e) => e.stopPropagation()}
                  autoFocus
                  className="text-xs px-1.5 py-0.5 bg-surface-raised border border-surface-border rounded text-gray-100 focus:outline-none focus:border-blue-400"
                />
              ) : (
                <span
                  onDoubleClick={() => handleTabDoubleClick(convo.id, convo.name)}
                  className="text-xs hover:text-gray-100 transition-colors"
                >
                  {convo.name}
                </span>
              )}
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  deleteConversation(convo.id)
                }}
                className="ml-1 p-0.5 hover:bg-surface-raised rounded transition-colors text-gray-500 hover:text-red-400"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          ))}

          <button
            onClick={handleCreateConversation}
            className="px-2.5 py-1.5 text-xs text-gray-400 hover:text-gray-100 rounded-lg hover:bg-surface-raised transition-colors whitespace-nowrap"
            title="Create new conversation"
          >
            + New
          </button>
        </div>
      )}

      {/* Repo/Snapshot selectors */}
      {conversations.length === 0 || !activeConvo ? (
        <div className="px-6 py-4 border-b border-surface-border space-y-3 bg-surface-overlay/50">
          <div className="space-y-2">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Repository</label>
              <select
                value={selectedRepoId}
                onChange={(e) => setSelectedRepoId(e.target.value)}
                className="w-full px-3 py-2 bg-surface-overlay border border-surface-border rounded text-sm text-gray-200 focus:outline-none focus:border-blue-500"
              >
                <option value="">Select a repository...</option>
                {repos.map((repo) => (
                  <option key={repo.id} value={repo.id}>
                    {repo.name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">Snapshot</label>
              <select
                value={selectedSnapshotId}
                onChange={(e) => setSelectedSnapshotId(e.target.value)}
                disabled={loadingSnapshots || snapshots.length === 0}
                className="w-full px-3 py-2 bg-surface-overlay border border-surface-border rounded text-sm text-gray-200 focus:outline-none focus:border-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <option value="">
                  {loadingSnapshots
                    ? 'Loading snapshots...'
                    : snapshots.length === 0
                    ? 'No snapshots available'
                    : 'Select a snapshot...'}
                </option>
                {snapshots.map((snap) => (
                  <option key={snap.id} value={snap.id}>
                    {snap.branch || 'unknown'} ({snap.id.slice(0, 8)})
                    {snap.status === 'ready' && ' ✓'}
                  </option>
                ))}
              </select>
            </div>

            <button
              onClick={handleCreateConversation}
              disabled={!selectedRepoId || !selectedSnapshotId}
              className="w-full px-4 py-2 bg-blue-700 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm font-medium"
            >
              Start a new conversation
            </button>
          </div>
        </div>
      ) : (
        <div className="border-b border-surface-border bg-surface-overlay/30">
          {/* Compact badge row */}
          <div className="px-6 py-2 flex items-center justify-between">
            <span className="text-xs text-gray-400">
              📁 {repos.find((r) => r.id === activeConvo.repoId)?.name || 'Unknown'} /{' '}
              {activeConvo.snapshotId.slice(0, 8)}
            </span>
            <button
              onClick={() => {
                setSelectedRepoId(activeConvo.repoId)
                setSelectedSnapshotId(activeConvo.snapshotId)
                setShowChangeSelector((v) => !v)
              }}
              className="text-xs text-gray-500 hover:text-blue-400 transition-colors"
            >
              {showChangeSelector ? 'Cancel' : 'Change'}
            </button>
          </div>

          {/* Expandable selector */}
          {showChangeSelector && (
            <div className="px-6 pb-3 space-y-2">
              <div className="flex gap-2">
                <div className="flex-1">
                  <label className="block text-xs text-gray-400 mb-1">Repository</label>
                  <select
                    value={selectedRepoId}
                    onChange={(e) => setSelectedRepoId(e.target.value)}
                    className="w-full px-2 py-1.5 bg-surface-overlay border border-surface-border rounded text-sm text-gray-200 focus:outline-none focus:border-blue-500"
                  >
                    {repos.map((repo) => (
                      <option key={repo.id} value={repo.id}>{repo.name}</option>
                    ))}
                  </select>
                </div>
                <div className="flex-1">
                  <label className="block text-xs text-gray-400 mb-1">Snapshot</label>
                  <select
                    value={selectedSnapshotId}
                    onChange={(e) => setSelectedSnapshotId(e.target.value)}
                    disabled={loadingSnapshots || snapshots.length === 0}
                    className="w-full px-2 py-1.5 bg-surface-overlay border border-surface-border rounded text-sm text-gray-200 focus:outline-none focus:border-blue-500 disabled:opacity-50"
                  >
                    {snapshots.map((snap) => (
                      <option key={snap.id} value={snap.id}>
                        {snap.branch || 'unknown'} ({snap.id.slice(0, 8)}){snap.status === 'ready' ? ' ✓' : ''}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <button
                onClick={handleConfirmChange}
                disabled={!selectedSnapshotId || !selectedRepoId}
                className="px-3 py-1.5 bg-blue-700 text-white text-xs rounded hover:bg-blue-600 disabled:opacity-50 transition-colors"
              >
                Apply
              </button>
            </div>
          )}
        </div>
      )}

      {/* Error banner */}
      {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

      {/* Messages area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {conversations.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-400">
              <p className="mb-4">No conversations yet</p>
              <p className="text-xs text-gray-500">Select a repository and snapshot above to get started</p>
            </div>
          </div>
        ) : !activeConvo ? (
          <div className="text-center text-gray-400 py-12">
            <p>Select a conversation to continue</p>
          </div>
        ) : messages.length === 0 ? (
          <div className="text-center text-gray-400 py-12">
            <p>Ask a question about the codebase to get started</p>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`max-w-2xl rounded-lg px-4 py-3 ${
                  msg.role === 'user'
                    ? 'bg-blue-700 text-gray-100'
                    : 'bg-surface-overlay text-gray-200'
                }`}
              >
                {msg.error ? (
                  <p className="text-red-400">{msg.error}</p>
                ) : (
                  <>
                    <div className="prose-qa">
                      <ReactMarkdown
                        components={{
                          h1: ({ children }) => <h1 className="text-base font-bold text-gray-100 mt-3 mb-1">{children}</h1>,
                          h2: ({ children }) => <h2 className="text-sm font-bold text-gray-100 mt-3 mb-1">{children}</h2>,
                          h3: ({ children }) => <h3 className="text-sm font-semibold text-gray-200 mt-2 mb-1">{children}</h3>,
                          h4: ({ children }) => <h4 className="text-xs font-semibold text-gray-300 mt-2 mb-0.5">{children}</h4>,
                          p: ({ children }) => <p className="text-sm text-gray-200 mb-2 last:mb-0 leading-relaxed">{children}</p>,
                          ul: ({ children }) => <ul className="list-disc list-inside text-sm text-gray-200 mb-2 space-y-0.5 pl-1">{children}</ul>,
                          ol: ({ children }) => <ol className="list-decimal list-inside text-sm text-gray-200 mb-2 space-y-0.5 pl-1">{children}</ol>,
                          li: ({ children }) => <li className="text-sm text-gray-200">{children}</li>,
                          strong: ({ children }) => <strong className="font-semibold text-gray-100">{children}</strong>,
                          em: ({ children }) => <em className="italic text-gray-300">{children}</em>,
                          hr: () => <hr className="border-surface-border my-2" />,
                          code: ({ children, className }) => {
                            const isBlock = className?.includes('language-')
                            return isBlock ? (
                              <code className="block bg-zinc-900 border border-zinc-700 rounded px-3 py-2 text-xs font-mono text-green-300 overflow-x-auto whitespace-pre">{children}</code>
                            ) : (
                              <code className="bg-zinc-800 px-1 py-0.5 rounded text-xs font-mono text-green-300">{children}</code>
                            )
                          },
                          pre: ({ children }) => <pre className="mb-2">{children}</pre>,
                          blockquote: ({ children }) => <blockquote className="border-l-2 border-blue-500 pl-3 my-2 text-gray-400 italic text-sm">{children}</blockquote>,
                        }}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                    {msg.response && (
                      <div className="mt-3 pt-3 border-t border-surface-border space-y-2">
                        {/* Confidence badge */}
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-gray-400">Confidence:</span>
                          <span
                            className={`text-xs font-semibold px-2 py-1 rounded ${
                              msg.response.confidence === 'high'
                                ? 'bg-green-900 text-green-200'
                                : msg.response.confidence === 'medium'
                                ? 'bg-yellow-900 text-yellow-200'
                                : 'bg-red-900 text-red-200'
                            }`}
                          >
                            {msg.response.confidence.toUpperCase()}
                          </span>
                        </div>

                        {/* Citations */}
                        {msg.response.citations.length > 0 && (
                          <div>
                            <p className="text-xs text-gray-400 mb-1">Citations:</p>
                            <div className="flex flex-wrap gap-1">
                              {msg.response.citations.map((cit, i) => (
                                <button
                                  key={i}
                                  onClick={() => handleCitationClick(cit.file, cit.line_start)}
                                  className="text-xs bg-surface-raised px-2 py-1 rounded cursor-pointer hover:bg-blue-900 transition-colors text-left"
                                  title={cit.snippet || 'Open in Snapshot Viewer'}
                                >
                                  📎 {cit.file}
                                  {cit.line_start ? `:${cit.line_start}` : ''}
                                </button>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Unknowns */}
                        {msg.response.unknowns.length > 0 && (
                          <div>
                            <p className="text-xs text-gray-400 mb-1">Unknown:</p>
                            <ul className="text-xs text-gray-300 list-disc list-inside space-y-0.5">
                              {msg.response.unknowns.map((uk, i) => (
                                <li key={i}>{uk}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Suggested files */}
                        {msg.response.suggested_files.length > 0 && (
                          <div>
                            <p className="text-xs text-gray-400 mb-1">Suggested files:</p>
                            <div className="flex flex-wrap gap-1">
                              {msg.response.suggested_files.map((file, i) => (
                                <span
                                  key={i}
                                  className="text-xs bg-surface-raised px-2 py-1 rounded cursor-pointer hover:bg-blue-900 transition-colors"
                                >
                                  {file}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          ))
        )}

        {/* Loading state */}
        {loading && (
          <div className="flex justify-start">
            <div className="flex items-center gap-2 bg-surface-overlay px-4 py-3 rounded-lg">
              <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
              <span className="text-sm text-gray-300">Thinking...</span>
            </div>
          </div>
        )}
      </div>

      {/* Input area */}
      {activeConvo && (
        <div className="px-6 py-4 border-t border-surface-border bg-surface flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about the codebase..."
            disabled={loading}
            className="flex-1 px-4 py-2 bg-surface-overlay border border-surface-border rounded-lg text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500 disabled:opacity-50"
          />
          <button
            onClick={handleAsk}
            disabled={loading || !input.trim()}
            className="px-4 py-2 bg-blue-700 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            Ask
          </button>
        </div>
      )}
    </div>
  )
}
