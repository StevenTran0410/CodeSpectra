import React from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import {
  Home,
  Cpu,
  GitBranch,
  FolderOpen,
  FlaskConical,
  FileText,
  Settings,
  ChevronDown,
  Plus
} from 'lucide-react'
import { useWorkspaceStore } from '../../store/workspace.store'
import { useState } from 'react'
import { WorkspaceModal } from '../workspace/WorkspaceModal'

const NAV_ITEMS = [
  { path: '/', label: 'Home', icon: Home, end: true },
  { path: '/providers', label: 'Providers', icon: Cpu },
  { path: '/code-hosts', label: 'Code Hosts', icon: GitBranch },
  { path: '/repositories', label: 'Repositories', icon: FolderOpen },
  { path: '/analysis', label: 'Analysis', icon: FlaskConical },
  { path: '/reports', label: 'Reports', icon: FileText }
]

export function Sidebar(): React.ReactElement {
  const { workspaces, activeWorkspaceId, setActive, create } = useWorkspaceStore()
  const [showWorkspacePicker, setShowWorkspacePicker] = useState(false)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const navigate = useNavigate()

  const activeWorkspace = workspaces.find((w) => w.id === activeWorkspaceId)

  return (
    <>
      <aside className="w-52 shrink-0 flex flex-col bg-surface border-r border-surface-border">
        {/* App header */}
        <div className="h-12 flex items-center px-4 border-b border-surface-border drag-region">
          <span className="font-semibold text-gray-100 text-sm no-drag">CodeSpectra</span>
        </div>

        {/* Workspace picker */}
        <div className="px-2 py-2 border-b border-surface-border">
          <button
            className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-surface-overlay transition-colors text-left"
            onClick={() => setShowWorkspacePicker((v) => !v)}
          >
            <div className="w-5 h-5 rounded bg-blue-600 flex items-center justify-center shrink-0">
              <span className="text-white text-xs font-bold leading-none">
                {activeWorkspace?.name?.[0]?.toUpperCase() ?? '?'}
              </span>
            </div>
            <span className="text-sm text-gray-200 truncate flex-1 min-w-0">
              {activeWorkspace?.name ?? 'Select workspace'}
            </span>
            <ChevronDown className="w-3.5 h-3.5 text-gray-500 shrink-0" />
          </button>

          {showWorkspacePicker && (
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
            >
              <Icon className="w-4 h-4 shrink-0" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Settings at bottom */}
        <div className="px-2 py-2 border-t border-surface-border">
          <NavLink
            to="/settings"
            className={({ isActive }) =>
              `flex items-center gap-2.5 px-2.5 py-2 rounded-md text-sm transition-colors ${
                isActive
                  ? 'bg-surface-overlay text-gray-100'
                  : 'text-gray-400 hover:bg-surface-overlay hover:text-gray-200'
              }`
            }
          >
            <Settings className="w-4 h-4 shrink-0" />
            Settings
          </NavLink>
        </div>
      </aside>

      {showCreateModal && (
        <WorkspaceModal
          mode="create"
          onConfirm={(name) => create(name).then(() => {})}
          onClose={() => setShowCreateModal(false)}
        />
      )}
    </>
  )
}
