import React, { useState, useRef, useEffect, useCallback } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { Send, Loader2, X, Microscope, Pencil, Copy, Check } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ErrorBanner } from '../../components/ui/ErrorBanner'
import { toErrorMessage } from '../../lib/errors'
import { useLocalRepoStore } from '../../store/local-repo.store'
import { useWorkspaceStore } from '../../store/workspace.store'
import { useAskScreen } from '../../hooks/useAskScreen'
import type { RepoSnapshot, QAResponse, DeepResearchResponse } from '../../types/electron'
import type { QAChatMessage } from '../../store/qa.store'

// Shared ReactMarkdown component map — used by both quick ask and deep research
const MD_COMPONENTS: React.ComponentProps<typeof ReactMarkdown>['components'] = {
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
    const content = String(children ?? '')
    // Block: has a language class OR is multiline (fenced block without explicit language)
    const isBlock = !!className || content.includes('\n')
    return isBlock ? (
      <code className="block bg-zinc-900 border border-zinc-700 rounded px-3 py-2 text-xs font-mono text-green-300 overflow-x-auto whitespace-pre">{children}</code>
    ) : (
      <code className="bg-zinc-800 px-1 py-0.5 rounded text-xs font-mono text-green-300">{children}</code>
    )
  },
  pre: ({ children }) => <pre className="mb-2 overflow-x-auto">{children}</pre>,
  blockquote: ({ children }) => <blockquote className="border-l-2 border-blue-500 pl-3 my-2 text-gray-400 italic text-sm">{children}</blockquote>,
  // ── Tables (remark-gfm) ──────────────────────────────────────────────────
  table: ({ children }) => (
    <div className="overflow-x-auto mb-3">
      <table className="w-full text-xs border-collapse">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-surface-raised">{children}</thead>,
  tbody: ({ children }) => <tbody>{children}</tbody>,
  tr: ({ children }) => <tr className="border-b border-surface-border">{children}</tr>,
  th: ({ children }) => (
    <th className="px-3 py-1.5 text-left font-semibold text-gray-300 border border-surface-border whitespace-nowrap">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="px-3 py-1.5 text-gray-200 border border-surface-border">{children}</td>
  ),
}

function DeepResearchConfirmCard({
  onConfirm,
  onSkip,
  onDisableForChat,
}: {
  onConfirm: () => void
  onSkip: () => void
  onDisableForChat: () => void
}): React.ReactElement {
  return (
    <div className="animate-in fade-in slide-in-from-bottom-2 duration-300 border border-purple-500/30 bg-purple-950/20 rounded-lg p-3 mt-2">
      <div className="flex items-center gap-2 text-purple-300 text-sm font-medium">
        <Microscope size={14} />
        Deep Research available
      </div>
      <p className="text-xs text-zinc-400 mt-1">
        This question may benefit from a deeper investigation across the codebase.
      </p>
      <div className="flex gap-2 mt-3">
        <button
          onClick={onConfirm}
          className="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 text-white text-xs rounded transition-colors"
        >
          Run Deep Research
        </button>
        <button
          onClick={onSkip}
          className="px-3 py-1.5 text-zinc-400 hover:text-zinc-200 text-xs rounded transition-colors"
        >
          Skip
        </button>
        <button
          onClick={onDisableForChat}
          className="px-3 py-1.5 text-zinc-500 hover:text-zinc-300 text-xs rounded transition-colors"
        >
          Don't ask again this chat
        </button>
      </div>
    </div>
  )
}

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
    truncateMessages,
    setActive,
    disableDeepResearch,
    isDeepResearchDisabled,
  } = useAskScreen()

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
  const [deepResearchPrompt, setDeepResearchPrompt] = useState<{
    conversationId: string
    snapshotId: string
    question: string
    messageIndex: number
  } | null>(null)
  const [forceDeepResearch, setForceDeepResearch] = useState(false)
  const [editingFromIdx, setEditingFromIdx] = useState<number | null>(null)
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null)
  // Streaming state
  const [streamPhase, setStreamPhase] = useState<string | null>(null)
  const [liveContent, setLiveContent] = useState<string>('')
  const [isStalled, setIsStalled] = useState(false)
  // Token buffer for rAF flush — accumulates tokens without triggering re-renders per token
  const tokenBufferRef = useRef<string[]>([])
  const rafIdRef = useRef<number>(0)
  // Increments on each new stream; stale rAF loops check against this to self-terminate
  const streamTagRef = useRef<number>(0)
  // Tracks last token timestamp to detect stalls (3s+ no tokens)
  const lastTokenAtRef = useRef<number>(0)
  // Guards stall state transitions to prevent re-render loops
  const prevIsStalledRef = useRef(false)
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

  // Cancel rAF loop on unmount
  useEffect(() => () => {
    cancelAnimationFrame(rafIdRef.current)
  }, [])

  /**
   * Start the rAF token flush loop for a specific stream tag.
   * Drains tokenBufferRef into liveContent once per animation frame.
   * Checks for token stalls (3s+ with no tokens) and guards state updates.
   * Self-terminates when the tag no longer matches (stream was superseded).
   */
  const startFlushLoop = useCallback((myTag: number) => {
    const flushLoop = () => {
      if (streamTagRef.current !== myTag) return
      const buf = tokenBufferRef.current.splice(0)
      if (buf.length > 0) {
        setLiveContent((prev) => prev + buf.join(''))
      }

      // Check for stall: 3 seconds since last token
      const now = Date.now()
      const timeSinceLastToken = now - lastTokenAtRef.current
      const shouldBeStalled = timeSinceLastToken > 3000

      // Only update state if stall status changed, avoiding re-render loop
      if (shouldBeStalled && !prevIsStalledRef.current) {
        prevIsStalledRef.current = true
        setIsStalled(true)
      } else if (!shouldBeStalled && prevIsStalledRef.current) {
        prevIsStalledRef.current = false
        setIsStalled(false)
      }

      rafIdRef.current = requestAnimationFrame(flushLoop)
    }
    rafIdRef.current = requestAnimationFrame(flushLoop)
  }, [])

  /** Shared teardown for a stream that ended (done or error). Flushes remaining tokens. */
  const stopStream = useCallback((handler: Parameters<typeof window.api.qa.offStreamEvent>[0]) => {
    window.api.qa.offStreamEvent(handler)
    cancelAnimationFrame(rafIdRef.current)
    const remaining = tokenBufferRef.current.splice(0)
    if (remaining.length > 0) {
      setLiveContent((prev) => prev + remaining.join(''))
    }
    setLoading(false)
    setStreamPhase(null)
    setIsStalled(false)
    prevIsStalledRef.current = false
  }, [])

  /** Shared stream setup: increments the tag, clears buffers, starts the flush loop. */
  const beginStream = useCallback(() => {
    streamTagRef.current++
    tokenBufferRef.current = []
    lastTokenAtRef.current = Date.now()
    setLiveContent('')
    setIsStalled(false)
    prevIsStalledRef.current = false
    startFlushLoop(streamTagRef.current)
    return streamTagRef.current
  }, [startFlushLoop])

  /**
   * Run quick ask via SSE stream. Tokens arrive via 'token' events and are flushed
   * into the live bubble once per requestAnimationFrame (zero re-renders per token).
   */
  const runAskStream = useCallback(async (convoId: string, snapshotId: string, question: string) => {
    const cfg = getConfig()
    const myTag = beginStream()

    return new Promise<void>((resolve) => {
      const streamHandler = (_ev: unknown, raw: unknown) => {
        const ev = raw as { type: string; phase?: string; text?: string; result?: QAResponse; message?: string }
        if (ev.type === 'status') {
          setStreamPhase(ev.phase ?? null)
        } else if (ev.type === 'token') {
          if (streamTagRef.current !== myTag) return
          // Switch to streaming phase — keeps loading=true to block double-submission
          setStreamPhase('streaming')
          tokenBufferRef.current.push(ev.text ?? '')
          // Update last token timestamp to track stalls
          lastTokenAtRef.current = Date.now()
        } else if (ev.type === 'done' && ev.result) {
          stopStream(streamHandler)
          const result = ev.result
          // Give one frame for final content to paint, then commit to message history
          requestAnimationFrame(() => {
            addMessage(convoId, { role: 'assistant', content: result.answer, response: result })
            setLiveContent('')
            resolve()
          })
        } else if (ev.type === 'error') {
          stopStream(streamHandler)
          const msg = (ev.message as string) || 'Unknown error'
          addMessage(convoId, { role: 'assistant', content: '', error: msg })
          setError(msg)
          setLiveContent('')
          resolve()
        }
      }
      window.api.qa.onStreamEvent(streamHandler)
      window.api.qa.askStream({
        snapshot_id: snapshotId,
        question,
        provider_id: cfg.providerId,
        model_id: cfg.modelId,
      })
    })
  }, [addMessage, beginStream, stopStream])

  /**
   * Run deep research via SSE stream. Emits planning/retrieving/thinking/synthesizing status
   * events, then streams the synthesis markdown token by token into the live bubble.
   */
  const runDeepResearchStream = useCallback(async (convoId: string, snapshotId: string, question: string) => {
    const cfg = getConfig()
    const myTag = beginStream()

    return new Promise<void>((resolve) => {
      const streamHandler = (_ev: unknown, raw: unknown) => {
        const ev = raw as { type: string; phase?: string; text?: string; result?: DeepResearchResponse; message?: string }
        if (ev.type === 'status') {
          setStreamPhase(ev.phase ?? null)
        } else if (ev.type === 'token') {
          if (streamTagRef.current !== myTag) return
          setStreamPhase('streaming')
          tokenBufferRef.current.push(ev.text ?? '')
          // Update last token timestamp to track stalls
          lastTokenAtRef.current = Date.now()
        } else if (ev.type === 'done' && ev.result) {
          stopStream(streamHandler)
          const result = ev.result
          requestAnimationFrame(() => {
            addMessage(convoId, {
              role: 'assistant',
              content: result.summary,
              deepResearchResponse: result,
            })
            setLiveContent('')
            resolve()
          })
        } else if (ev.type === 'error') {
          stopStream(streamHandler)
          const msg = (ev.message as string) || 'Unknown error'
          addMessage(convoId, { role: 'assistant', content: '', error: msg })
          setError(msg)
          setLiveContent('')
          resolve()
        }
      }
      window.api.qa.onStreamEvent(streamHandler)
      window.api.qa.deepResearchStream({
        snapshot_id: snapshotId,
        question,
        provider_id: cfg.providerId,
        model_id: cfg.modelId,
      })
    })
  }, [addMessage, beginStream, stopStream])

  // Auto-scroll to bottom on new messages or live content changes
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
    }
  }, [messages, liveContent])

  // Clear deepResearchPrompt on conversation change
  useEffect(() => {
    setDeepResearchPrompt(null)
  }, [activeId])

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
    setDeepResearchPrompt(null)

    // When editing an existing message, truncate history from that index
    const baseMessageIndex = editingFromIdx ?? (activeConvo.messages?.length ?? 0)
    if (editingFromIdx !== null) {
      truncateMessages(activeId, editingFromIdx)
      setEditingFromIdx(null)
    }

    addMessage(activeId, { role: 'user', content: q })
    setLoading(true)
    setError(null)

    try {
      if (!providerId || !modelId) {
        throw new Error('No provider/model configured. Please set up a provider in the Providers screen.')
      }

      // 1. Determine mode — manual override takes priority, then local classifier
      let isDeep = forceDeepResearch
      if (!isDeep && !isDeepResearchDisabled(activeId)) {
        try {
          const classification = await window.api.qa.classifyIntent({ question: q })
          isDeep = classification.deep_research
        } catch {
          // classify failed — fall through to quick ask
        }
      }

      if (isDeep) {
        // Stop here — show confirm card unless manual toggle (then skip confirm)
        setLoading(false)
        if (forceDeepResearch) {
          // Manual toggle: skip confirm card, run deep research directly
          setForceDeepResearch(false)
          setLoading(true)
          await runDeepResearchStream(activeId, activeConvo.snapshotId, q)
          return
        }
        // Auto-detected: show confirm card
        setDeepResearchPrompt({
          conversationId: activeId,
          snapshotId: activeConvo.snapshotId,
          question: q,
          messageIndex: baseMessageIndex,
        })
        return
      }

      // 2. Not deep research — run quick ask via stream
      await runAskStream(activeId, activeConvo.snapshotId, q)
    } catch (err) {
      const msg = toErrorMessage(err)
      addMessage(activeId, { role: 'assistant', content: '', error: msg })
      setError(msg)
      setLoading(false)
    }
  }

  const handleDeepResearchConfirm = async () => {
    if (!deepResearchPrompt) return
    const { conversationId, snapshotId, question } = deepResearchPrompt
    setDeepResearchPrompt(null)
    setLoading(true)
    setError(null)
    await runDeepResearchStream(conversationId, snapshotId, question)
  }

  const handleDeepResearchSkip = async () => {
    if (!deepResearchPrompt) return
    const { conversationId, snapshotId, question } = deepResearchPrompt
    setDeepResearchPrompt(null)
    setLoading(true)
    setError(null)
    await runAskStream(conversationId, snapshotId, question)
  }

  const handleDisableForChat = async () => {
    if (!deepResearchPrompt) return
    const { conversationId } = deepResearchPrompt
    disableDeepResearch(conversationId)
    // Also run quick ask since user is dismissing deep research permanently for this chat
    await handleDeepResearchSkip()
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
    <div className="h-full flex flex-col bg-app-bg overflow-hidden">
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
                aria-label="Delete conversation"
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
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-4 space-y-6">
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
            <React.Fragment key={idx}>
              <div className={`flex group ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className="relative max-w-2xl">
                  {/* Action buttons — hover reveal */}
                  <div className={`absolute -top-6 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity ${msg.role === 'user' ? 'right-0' : 'left-0'}`}>
                    {msg.role === 'user' && (
                      <button
                        onClick={() => {
                          setInput(msg.content)
                          setEditingFromIdx(idx)
                        }}
                        title="Edit and resend"
                        className="p-1 rounded text-gray-500 hover:text-gray-200 hover:bg-surface-overlay transition-colors"
                      >
                        <Pencil size={12} />
                      </button>
                    )}
                    <button
                      onClick={() => {
                        navigator.clipboard.writeText(msg.content)
                        setCopiedIdx(idx)
                        setTimeout(() => setCopiedIdx((v) => (v === idx ? null : v)), 1500)
                      }}
                      title="Copy text"
                      className="p-1 rounded text-gray-500 hover:text-gray-200 hover:bg-surface-overlay transition-colors"
                    >
                      {copiedIdx === idx ? <Check size={12} className="text-green-400" /> : <Copy size={12} />}
                    </button>
                  </div>
                <div
                  className={`rounded-lg px-4 py-3 ${
                    msg.role === 'user'
                      ? 'bg-blue-700 text-gray-100'
                      : 'bg-surface-overlay text-gray-200'
                  } ${editingFromIdx === idx ? 'ring-2 ring-yellow-400/60' : ''}`}
                >
                {msg.error ? (
                  <p className="text-red-400 select-text">{msg.error}</p>
                ) : (
                  <>
                    <div className="prose-qa select-text">
                      <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                    {msg.deepResearchResponse && (
                      <DeepResearchResultView
                        result={msg.deepResearchResponse}
                        onCitationClick={handleCitationClick}
                      />
                    )}
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
                                  className="text-xs bg-surface-raised px-2 py-1 rounded cursor-pointer hover:bg-blue-900 transition-colors text-left truncate max-w-[200px]"
                                  title={cit.file}
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
            </div>
            {deepResearchPrompt &&
              deepResearchPrompt.conversationId === activeId &&
              deepResearchPrompt.messageIndex === idx && (
                <div className="flex justify-start">
                  <div className="max-w-2xl w-full">
                    <DeepResearchConfirmCard
                      onConfirm={handleDeepResearchConfirm}
                      onSkip={handleDeepResearchSkip}
                      onDisableForChat={handleDisableForChat}
                    />
                  </div>
                </div>
              )}
            </React.Fragment>
          ))
        )}

        {/* Loading state — granular phase label */}
        {loading && streamPhase !== 'streaming' && (
          <div className="flex justify-start">
            <div className="flex items-center gap-2 px-4 py-3 rounded-lg bg-surface-overlay">
              <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
              <span className="text-sm text-gray-300">
                {streamPhase === 'retrieving'
                  ? 'Retrieving...'
                  : streamPhase === 'planning'
                  ? 'Planning...'
                  : streamPhase === 'synthesizing'
                  ? 'Synthesizing...'
                  : 'Thinking...'}
              </span>
            </div>
          </div>
        )}

        {/* Live streaming bubble — tokens flush in via rAF */}
        {liveContent.length > 0 && (
          <div className="flex justify-start">
            <div className="relative max-w-2xl">
              <div className="rounded-lg px-4 py-3 bg-surface-overlay text-gray-200">
                <div className="prose-qa select-text">
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
                    {liveContent}
                  </ReactMarkdown>
                </div>
                <span className="inline-block w-1.5 h-4 bg-blue-400 animate-pulse ml-0.5 align-text-bottom rounded-sm" />
                {isStalled && (
                  <span className="text-xs text-yellow-300 block mt-1">(waiting for response…)</span>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input area */}
      {activeConvo && (
        <div className="px-6 py-4 border-t border-surface-border bg-surface space-y-2">
          {/* Editing indicator */}
          {editingFromIdx !== null && (
            <div className="flex items-center gap-2 text-xs text-yellow-400">
              <Pencil size={11} />
              Editing message — resend to replace from this point
              <button
                onClick={() => { setEditingFromIdx(null); setInput('') }}
                className="ml-auto text-gray-500 hover:text-gray-300 transition-colors"
              >
                <X size={12} />
              </button>
            </div>
          )}
          {/* Toolbar */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => setForceDeepResearch((v) => !v)}
              title={forceDeepResearch ? 'Deep Research ON — click to disable' : 'Enable Deep Research for next message'}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                forceDeepResearch
                  ? 'bg-purple-600 text-white hover:bg-purple-500'
                  : 'text-zinc-500 hover:text-zinc-300 hover:bg-surface-overlay'
              }`}
            >
              <Microscope size={12} />
              Deep Research
            </button>
          </div>
          {/* Input field */}
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={forceDeepResearch ? 'Deep Research mode — ask anything...' : 'Ask a question about the codebase...'}
              disabled={loading}
              className={`flex-1 px-4 py-2 bg-surface-overlay border rounded-lg text-gray-100 placeholder-gray-500 focus:outline-none disabled:opacity-50 transition-colors ${
                forceDeepResearch ? 'border-purple-500/60 focus:border-purple-400' : 'border-surface-border focus:border-blue-500'
              }`}
            />
            <button
              onClick={handleAsk}
              disabled={loading || !input.trim()}
              className={`px-4 py-2 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2 ${
                forceDeepResearch ? 'bg-purple-700 hover:bg-purple-600' : 'bg-blue-700 hover:bg-blue-600'
              }`}
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : (forceDeepResearch ? <Microscope className="w-4 h-4" /> : <Send className="w-4 h-4" />)}
              {loading
                ? streamPhase === 'retrieving' ? 'Retrieving…'
                  : streamPhase === 'planning' ? 'Planning…'
                  : streamPhase === 'synthesizing' ? 'Synthesizing…'
                  : streamPhase === 'streaming' ? 'Streaming…'
                  : 'Thinking…'
                : forceDeepResearch ? 'Research'
                : editingFromIdx !== null ? 'Resend'
                : 'Ask'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function DeepResearchResultView({
  result,
  onCitationClick,
}: {
  result: DeepResearchResponse
  onCitationClick: (file: string, line?: number | null) => void
}): React.ReactElement {
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set([0]))

  return (
    <div className="mt-3 space-y-3">
      {/* Header bar */}
      <div className="flex items-center gap-3 text-xs text-gray-400">
        <span className="flex items-center gap-1">
          <Microscope className="w-3 h-3 text-purple-400" />
          Deep Research
        </span>
        <span>{(result.elapsed_ms / 1000).toFixed(1)}s</span>
        <span>{result.files_explored.length} files explored</span>
        <span
          className={`px-2 py-0.5 rounded font-semibold ${
            result.confidence === 'high'
              ? 'bg-green-900 text-green-200'
              : result.confidence === 'medium'
              ? 'bg-yellow-900 text-yellow-200'
              : 'bg-red-900 text-red-200'
          }`}
        >
          {result.confidence.toUpperCase()}
        </span>
      </div>

      {/* Investigation chain */}
      {result.reasoning_chain.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-semibold text-gray-400">Investigation Chain</p>
          {result.reasoning_chain.map((step) => (
            <div key={step.step_number} className="border-l-2 border-purple-700 pl-3 space-y-1">
              <button
                className="text-xs font-semibold text-gray-300 hover:text-gray-100 text-left"
                onClick={() =>
                  setExpandedSteps((prev) => {
                    const next = new Set(prev)
                    next.has(step.step_number) ? next.delete(step.step_number) : next.add(step.step_number)
                    return next
                  })
                }
              >
                Step {step.step_number}: {step.description}
              </button>
              {expandedSteps.has(step.step_number) && (
                <div className="space-y-1">
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
                    {step.finding}
                  </ReactMarkdown>
                  {step.graph_path && step.graph_path.length > 1 && (
                    <p className="text-xs text-purple-300 font-mono">{step.graph_path.join(' → ')}</p>
                  )}
                  {step.files_involved.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {step.files_involved.map((file, i) => (
                        <button
                          key={i}
                          onClick={() => onCitationClick(file)}
                          className="text-xs bg-surface-raised px-2 py-0.5 rounded hover:bg-purple-900 transition-colors"
                        >
                          {file}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Unknowns */}
      {result.unknowns.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-400 mb-1">Unknowns</p>
          <ul className="text-xs text-gray-400 list-disc list-inside space-y-0.5">
            {result.unknowns.map((u, i) => (
              <li key={i}>{u}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
