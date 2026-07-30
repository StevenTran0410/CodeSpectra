import React, { useState, useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { NavLink, useNavigate, useLocation } from 'react-router-dom'
import {
  Home,
  Cpu,
  GitBranch,
  FolderOpen,
  FlaskConical,
  FileText,
  MessageSquare,
  Settings,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Plus,
  Sun,
  Moon,
  Layers,
  Code
} from 'lucide-react'
import { useWorkspaceStore } from '../../store/workspace.store'
import { WorkspaceModal } from '../workspace/WorkspaceModal'
import { useTheme } from '../../hooks/useTheme'

const NAV_ITEMS = [
  { path: '/', label: 'Home', icon: Home, end: true },
  { path: '/providers', label: 'Providers', icon: Cpu },
  { path: '/code-hosts', label: 'Code Hosts', icon: GitBranch }
]

const CA_ITEMS = [
  { path: '/ca/repositories', label: 'Repositories', icon: FolderOpen },
  { path: '/ca/analysis', label: 'Analysis', icon: FlaskConical },
  { path: '/ca/reports', label: 'Reports', icon: FileText },
  { path: '/ca/ask', label: 'Ask', icon: MessageSquare }
]

const AEH_ITEMS = [
  { path: '/aeh/repositories', label: 'Repositories', icon: FolderOpen },
  { path: '/aeh/analysis', label: 'Analysis', icon: FlaskConical },
  { path: '/aeh/reports', label: 'Reports', icon: FileText },
  { path: '/aeh/ask', label: 'Ask', icon: MessageSquare }
]

const MODE_PREVIEWS = {
  ca: {
    title: 'Code Analysis',
    description: 'Explore & understand code — repository indexing, static analysis reports, and Q&A over your codebase.'
  },
  aeh: {
    title: 'Agentic Evaluation Harness',
    description: 'Discover and evaluate agentic systems in your codebase — traces, metrics, and per-component drill-down.'
  }
} as const

type ModeKey = keyof typeof MODE_PREVIEWS

const HOVER_DELAY_MS = 450

/** Renders full, untruncated mode info in a small floating card, positioned from a
 * trigger element's live bounding rect and portaled to <body> so it can pop out
 * past the sidebar's own `overflow-hidden` instead of being clipped by it. */
function ModeHoverPreview({ modeKey, anchorRect }: { modeKey: ModeKey; anchorRect: DOMRect }): React.ReactElement {
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    const id = requestAnimationFrame(() => setVisible(true))
    return () => cancelAnimationFrame(id)
  }, [])

  const info = MODE_PREVIEWS[modeKey]
  const style: React.CSSProperties = {
    position: 'fixed',
    top: anchorRect.bottom + 6,
    left: anchorRect.left,
    width: Math.max(anchorRect.width, 240),
    zIndex: 9999
  }

  return createPortal(
    <div
      style={style}
      className={`pointer-events-none rounded-lg border shadow-xl p-3 transition-all duration-150 origin-top ${
        visible ? 'opacity-100 scale-100' : 'opacity-0 scale-95'
      } ${
        modeKey === 'aeh'
          ? 'bg-surface-raised border-indigo-500/30'
          : 'bg-surface-raised border-surface-border'
      }`}
    >
      <div className={`text-sm font-semibold ${modeKey === 'aeh' ? 'text-white' : 'text-gray-100'}`}>
        {info.title}
      </div>
      <div className="text-xs text-gray-400 mt-1 leading-relaxed">{info.description}</div>
    </div>,
    document.body
  )
}

export function Sidebar(): React.ReactElement {
  const { workspaces, activeWorkspaceId, setActive, create } = useWorkspaceStore()
  const [showWorkspacePicker, setShowWorkspacePicker] = useState(false)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [isCollapsed, setIsCollapsed] = useState(() => localStorage.getItem('sidebar.collapsed') === '1')
  const navigate = useNavigate()
  const { isDark, toggle } = useTheme()
  const location = useLocation()

  const [isCAExpanded, setIsCAExpanded] = useState(() => {
    const saved = localStorage.getItem('sidebar.ca_expanded')
    return saved === null ? true : saved === '1'
  })
  const [isAEHExpanded, setIsAEHExpanded] = useState(() => {
    const saved = localStorage.getItem('sidebar.aeh_expanded')
    return saved === null ? false : saved === '1'
  })

  const [hoverPreview, setHoverPreview] = useState<{ key: ModeKey; rect: DOMRect } | null>(null)
  const hoverTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const caCardRef = useRef<HTMLButtonElement>(null)
  const aehCardRef = useRef<HTMLButtonElement>(null)

  const scheduleHoverPreview = (key: ModeKey, el: HTMLButtonElement | null) => {
    if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current)
    hoverTimeoutRef.current = setTimeout(() => {
      if (el) setHoverPreview({ key, rect: el.getBoundingClientRect() })
    }, HOVER_DELAY_MS)
  }
  const cancelHoverPreview = () => {
    if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current)
    setHoverPreview(null)
  }

  // Auto-expand active group on mount / navigation
  useEffect(() => {
    if (location.pathname.startsWith('/ca')) {
      setIsCAExpanded(true)
      localStorage.setItem('sidebar.ca_expanded', '1')
    } else if (location.pathname.startsWith('/aeh')) {
      setIsAEHExpanded(true)
      localStorage.setItem('sidebar.aeh_expanded', '1')
    }
  }, [location.pathname])

  const toggleCA = () => {
    const next = !isCAExpanded
    setIsCAExpanded(next)
    localStorage.setItem('sidebar.ca_expanded', next ? '1' : '0')
  }

  const toggleAEH = () => {
    const next = !isAEHExpanded
    setIsAEHExpanded(next)
    localStorage.setItem('sidebar.aeh_expanded', next ? '1' : '0')
  }

  const activeWorkspace = workspaces.find((w) => w.id === activeWorkspaceId)

  const toggleCollapse = () => {
    const newState = !isCollapsed
    setIsCollapsed(newState)
    localStorage.setItem('sidebar.collapsed', newState ? '1' : '0')
  }

  return (
    <>
      <aside className={`${isCollapsed ? 'w-14' : 'w-52'} shrink-0 flex flex-col bg-surface border-r border-surface-border transition-[width] duration-200 ease-in-out overflow-hidden`}>
        {/* App header */}
        <div className="h-12 flex items-center px-4 border-b border-surface-border drag-region">
          {!isCollapsed && <span className="font-semibold text-gray-100 text-sm no-drag">CodeSpectra</span>}
        </div>

        {/* Workspace picker */}
        <div className="px-2 py-2 border-b border-surface-border">
          <button
            className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-surface-overlay transition-colors text-left"
            onClick={() => setShowWorkspacePicker((v) => !v)}
            title={isCollapsed ? (activeWorkspace?.name ?? 'Select workspace') : undefined}
          >
            <div className="w-5 h-5 rounded bg-blue-600 flex items-center justify-center shrink-0">
              <span className="text-white text-xs font-bold leading-none">
                {activeWorkspace?.name?.[0]?.toUpperCase() ?? '?'}
              </span>
            </div>
            {!isCollapsed && (
              <>
                <span className="text-sm text-gray-200 truncate flex-1 min-w-0">
                  {activeWorkspace?.name ?? 'Select workspace'}
                </span>
                <ChevronDown className="w-3.5 h-3.5 text-gray-500 shrink-0" />
              </>
            )}
          </button>

          {showWorkspacePicker && !isCollapsed && (
            <div className="mt-1 rounded-md border border-surface-border bg-surface-overlay shadow-lg">
              {workspaces.map((ws) => (
                <button
                  key={ws.id}
                  className={`w-full text-left px-3 py-2 text-sm hover:bg-surface-raised transition-colors first:rounded-t-md ${
                    ws.id === activeWorkspaceId ? 'text-blue-400' : 'text-gray-300'
                  }`}
                  onClick={() => {
                    setActive(ws.id)
                    setShowWorkspacePicker(false)
                    navigate('/')
                  }}
                >
                  {ws.name}
                </button>
              ))}
              <button
                className="w-full text-left px-3 py-2 text-sm text-gray-500 hover:bg-surface-raised flex items-center gap-1.5 border-t border-surface-border last:rounded-b-md"
                onClick={() => {
                  setShowWorkspacePicker(false)
                  setShowCreateModal(true)
                }}
              >
                <Plus className="w-3.5 h-3.5" />
                New workspace
              </button>
            </div>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-2 py-2 space-y-0.5 overflow-y-auto">
          {NAV_ITEMS.map(({ path, label, icon: Icon, end }) => (
            <NavLink
              key={path}
              to={path}
              end={end}
              className={({ isActive }) =>
                `flex items-center gap-2.5 px-2.5 py-2 rounded-md text-sm transition-colors ${
                  isActive
                    ? 'bg-surface-overlay text-gray-100'
                    : 'text-gray-400 hover:bg-surface-overlay hover:text-gray-200'
                }`
              }
              title={isCollapsed ? label : undefined}
            >
              <Icon className="w-4 h-4 shrink-0" />
              {!isCollapsed && <span>{label}</span>}
            </NavLink>
          ))}
          
          <div className="border-t border-surface-border my-3 pt-3" />

          {!isCollapsed && (
            <div className="px-1 pb-1 text-[10px] font-semibold tracking-wider text-gray-500 uppercase">
              Workspace modes
            </div>
          )}

          {/* Code Analysis mode card */}
          <button
            ref={caCardRef}
            onClick={toggleCA}
            onMouseEnter={() => !isCollapsed && scheduleHoverPreview('ca', caCardRef.current)}
            onMouseLeave={cancelHoverPreview}
            className={`w-full flex items-start gap-2.5 px-2.5 py-2 rounded-lg text-sm transition-colors group ${
              isCollapsed ? 'justify-center' : ''
            } ${
              location.pathname.startsWith('/ca')
                ? 'bg-surface-overlay'
                : 'hover:bg-surface-overlay/60'
            }`}
            title={isCollapsed ? 'Code Analysis' : undefined}
          >
            <span className="flex items-center justify-center w-7 h-7 rounded-md bg-surface-raised border border-surface-border text-gray-300 shrink-0">
              <Code className="w-3.5 h-3.5" />
            </span>
            {!isCollapsed && (
              <>
                <span className="flex-1 min-w-0 text-left pt-0.5">
                  <span className="block font-semibold text-gray-100 truncate">Code Analysis</span>
                  <span className="block text-[11px] text-gray-500 truncate">Explore &amp; understand code</span>
                </span>
                <ChevronRight
                  className={`w-3.5 h-3.5 text-gray-500 shrink-0 mt-1.5 transition-transform duration-200 ${
                    isCAExpanded ? 'rotate-90' : ''
                  }`}
                />
              </>
            )}
          </button>

          {isCAExpanded && (
            <div className={`${!isCollapsed ? 'ml-[22px] pl-3 border-l border-surface-border' : ''} space-y-0.5 mt-1 mb-1`}>
              {CA_ITEMS.map(({ path, label, icon: Icon }) => (
                <NavLink
                  key={path}
                  to={path}
                  className={({ isActive }) =>
                    `flex items-center gap-2.5 px-2.5 py-1.5 rounded-md text-sm transition-colors ${
                      isActive
                        ? 'bg-surface-overlay text-gray-100 font-medium'
                        : 'text-gray-400 hover:bg-surface-overlay hover:text-gray-200'
                    }`
                  }
                  title={isCollapsed ? label : undefined}
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  {!isCollapsed && <span>{label}</span>}
                </NavLink>
              ))}
            </div>
          )}

          <div className="border-t border-surface-border my-2 pt-2" />

          {/* AEH mode card */}
          <button
            ref={aehCardRef}
            onClick={toggleAEH}
            onMouseEnter={() => !isCollapsed && scheduleHoverPreview('aeh', aehCardRef.current)}
            onMouseLeave={cancelHoverPreview}
            className={`w-full flex items-start gap-2.5 px-2.5 py-2 rounded-lg text-sm transition-all duration-200 border ${
              isCollapsed ? 'justify-center' : ''
            } ${
              location.pathname.startsWith('/aeh')
                ? 'bg-indigo-500/10 border-indigo-500/30'
                : 'border-transparent hover:bg-indigo-500/5 hover:border-indigo-500/10'
            }`}
            title={isCollapsed ? 'Agentic Eval Harness' : undefined}
          >
            <span className="flex items-center justify-center w-7 h-7 rounded-md bg-gradient-to-br from-indigo-500 to-purple-600 shadow-sm shadow-indigo-500/40 shrink-0">
              <Layers className="w-3.5 h-3.5 text-white" />
            </span>
            {!isCollapsed && (
              <>
                <span className="flex-1 min-w-0 text-left pt-0.5">
                  <span className="flex items-center gap-1.5">
                    <span className="font-bold text-white truncate">AEH</span>
                    <span className="px-1.5 py-[1px] rounded-full text-[9px] font-semibold uppercase tracking-wide bg-indigo-500/15 text-indigo-300 border border-indigo-500/20">
                      Beta
                    </span>
                  </span>
                  <span className="block text-[11px] text-indigo-300/70 truncate">Agentic Evaluation Harness</span>
                </span>
                <ChevronRight
                  className={`w-3.5 h-3.5 text-indigo-400 shrink-0 mt-1.5 transition-transform duration-200 ${
                    isAEHExpanded ? 'rotate-90' : ''
                  }`}
                />
              </>
            )}
          </button>

          {isAEHExpanded && (
            <div className={`${!isCollapsed ? 'ml-[22px] pl-3 border-l border-indigo-500/20' : ''} space-y-0.5 mt-1`}>
              {AEH_ITEMS.map(({ path, label, icon: Icon }) => (
                <NavLink
                  key={path}
                  to={path}
                  className={({ isActive }) =>
                    `flex items-center gap-2.5 px-2.5 py-1.5 rounded-md text-sm transition-colors ${
                      isActive
                        ? 'bg-indigo-500/10 text-white font-medium'
                        : 'text-indigo-300/80 hover:bg-indigo-500/5 hover:text-indigo-200'
                    }`
                  }
                  title={isCollapsed ? label : undefined}
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  {!isCollapsed && <span>{label}</span>}
                </NavLink>
              ))}
            </div>
          )}
        </nav>

        {/* Settings + theme toggle at bottom */}
        <div className="px-2 py-2 border-t border-surface-border space-y-0.5">
          <button
            className="btn-ghost w-full justify-start px-2.5 py-2 text-sm"
            aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            aria-expanded={!isCollapsed}
            onClick={toggleCollapse}
            title={isCollapsed ? 'Expand' : 'Collapse'}
          >
            {isCollapsed ? <ChevronRight className="w-4 h-4 shrink-0" /> : <ChevronLeft className="w-4 h-4 shrink-0" />}
            {!isCollapsed && (isCollapsed ? 'Expand' : 'Collapse')}
          </button>
          <NavLink
            to="/settings"
            className={({ isActive }) =>
              `flex items-center gap-2.5 px-2.5 py-2 rounded-md text-sm transition-colors ${
                isActive
                  ? 'bg-surface-overlay text-gray-100'
                  : 'text-gray-400 hover:bg-surface-overlay hover:text-gray-200'
              }`
            }
            title={isCollapsed ? 'Settings' : undefined}
          >
            <Settings className="w-4 h-4 shrink-0" />
            {!isCollapsed && <span>Settings</span>}
          </NavLink>
          <button
            className="btn-ghost w-full justify-start px-2.5 py-2 text-sm"
            aria-label="Toggle theme"
            onClick={toggle}
            title={isCollapsed ? (isDark ? 'Light mode' : 'Dark mode') : undefined}
          >
            {isDark ? <Sun className="w-4 h-4 shrink-0" /> : <Moon className="w-4 h-4 shrink-0" />}
            {!isCollapsed && (isDark ? 'Light mode' : 'Dark mode')}
          </button>
        </div>
      </aside>

      {hoverPreview && <ModeHoverPreview modeKey={hoverPreview.key} anchorRect={hoverPreview.rect} />}

      {showCreateModal && (
        <WorkspaceModal
          mode="create"
          onConfirm={(name, description) => create(name, description).then(() => {})}
          onClose={() => setShowCreateModal(false)}
        />
      )}
    </>
  )
}
