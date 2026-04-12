import React, { useState } from 'react'
import { FolderOpen, Plus, Pencil, Trash2, MoreHorizontal } from 'lucide-react'
import { useWorkspaceStore } from '../../store/workspace.store'
import { WorkspaceModal } from '../../components/workspace/WorkspaceModal'
import { EmptyState } from '../../components/ui/EmptyState'
import { CardSkeleton } from '../../components/ui/LoadingSkeleton'

type ModalState =
  | { type: 'none' }
  | { type: 'create' }
  | { type: 'rename'; id: string; name: string }
  | { type: 'delete'; id: string; name: string }

export default function Home(): React.ReactElement {
  const { workspaces, activeWorkspaceId, isLoading, setActive, create, rename, remove } =
    useWorkspaceStore()
  const [modal, setModal] = useState<ModalState>({ type: 'none' })
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null)

  if (isLoading) {
    return (
      <div className="p-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {[1, 2, 3].map((i) => (
          <CardSkeleton key={i} />
        ))}
      </div>
    )
  }

  if (workspaces.length === 0) {
    return (
      <div className="h-full">
        <EmptyState
          icon={<FolderOpen className="w-7 h-7" />}
          title="No workspaces yet"
          description="A workspace groups your repository connections, analysis runs, and reports."
          action={
            <button className="btn-primary" onClick={() => setModal({ type: 'create' })}>
              <Plus className="w-4 h-4" />
              Create workspace
            </button>
          }
        />
        {modal.type === 'create' && (
          <WorkspaceModal
            mode="create"
            onConfirm={(name, description) => create(name, description).then(() => {})}
            onClose={() => setModal({ type: 'none' })}
          />
        )}
      </div>
    )
  }

  return (
    <>
      <div className="screen-header flex items-center justify-between">
        <div>
          <h1 className="screen-title">Workspaces</h1>
          <p className="screen-subtitle">{workspaces.length} workspace{workspaces.length !== 1 ? 's' : ''}</p>
        </div>
        <button className="btn-primary" onClick={() => setModal({ type: 'create' })}>
          <Plus className="w-4 h-4" />
          New
        </button>
      </div>

      <div className="p-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {workspaces.map((ws) => (
          <div
            key={ws.id}
            className={`card p-4 cursor-pointer transition-colors hover:border-gray-600 relative group ${
              ws.id === activeWorkspaceId ? 'border-blue-700 bg-blue-950/20' : ''
            }`}
            onClick={() => setActive(ws.id)}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex items-center gap-3 min-w-0">
                <div className="w-8 h-8 rounded-lg bg-blue-600/20 border border-blue-700/30 flex items-center justify-center shrink-0">
                  <span className="text-blue-400 text-sm font-semibold">
                    {ws.name[0]?.toUpperCase()}
                  </span>
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-medium text-gray-100 truncate">{ws.name}</p>
                  {ws.description && (
                    <p className="text-xs text-gray-500 mt-0.5 truncate">{ws.description}</p>
                  )}
                  {!ws.description && (
                    <p className="text-xs text-gray-500 mt-0.5">
                      Created {new Date(ws.created_at).toLocaleDateString()}
                    </p>
                  )}
                </div>
              </div>

              <div className="relative shrink-0">
                <button
                  className="btn-ghost p-1 opacity-0 group-hover:opacity-100 hover:opacity-100"
                  onClick={(e) => {
                    e.stopPropagation()
                    setMenuOpenId(menuOpenId === ws.id ? null : ws.id)
                  }}
                >
                  <MoreHorizontal className="w-4 h-4" />
                </button>

                {menuOpenId === ws.id && (
                  <div className="absolute right-0 top-7 z-20 w-36 rounded-md border border-surface-border bg-surface-overlay shadow-lg py-1">
                    <button
                      className="w-full text-left px-3 py-2 text-sm text-gray-300 hover:bg-surface-raised flex items-center gap-2"
                      onClick={(e) => {
                        e.stopPropagation()
                        setMenuOpenId(null)
                        setModal({ type: 'rename', id: ws.id, name: ws.name })
                      }}
                    >
                      <Pencil className="w-3.5 h-3.5" />
                      Rename
                    </button>
                    <button
                      className="w-full text-left px-3 py-2 text-sm text-red-400 hover:bg-surface-raised flex items-center gap-2"
                      onClick={(e) => {
                        e.stopPropagation()
                        setMenuOpenId(null)
                        setModal({ type: 'delete', id: ws.id, name: ws.name })
                      }}
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                      Delete
                    </button>
                  </div>
                )}
              </div>
            </div>

            {ws.id === activeWorkspaceId && (
              <div className="mt-3 flex items-center gap-1.5 text-xs text-blue-400">
                <span className="h-1.5 w-1.5 rounded-full bg-blue-400" />
                Active
              </div>
            )}
          </div>
        ))}
      </div>

      {menuOpenId && (
        <div className="fixed inset-0 z-10" onClick={() => setMenuOpenId(null)} />
      )}

      {modal.type === 'create' && (
        <WorkspaceModal
          mode="create"
          onConfirm={(name, description) => create(name, description).then(() => {})}
          onClose={() => setModal({ type: 'none' })}
        />
      )}

      {modal.type === 'rename' && (
        <WorkspaceModal
          mode="rename"
          initialName={modal.name}
          onConfirm={(name) => rename(modal.id, name)}
          onClose={() => setModal({ type: 'none' })}
        />
      )}

      {modal.type === 'delete' && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
          onClick={(e) => e.target === e.currentTarget && setModal({ type: 'none' })}
        >
          <div className="card w-full max-w-sm p-5 shadow-2xl">
            <h2 className="text-base font-semibold text-gray-100">Delete workspace?</h2>
            <p className="text-sm text-gray-400 mt-2">
              <strong className="text-gray-200">{modal.name}</strong> will be permanently deleted.
              All repository connections and analysis history will be lost.
            </p>
            <div className="flex gap-2 justify-end mt-5">
              <button className="btn-secondary" onClick={() => setModal({ type: 'none' })}>
                Cancel
              </button>
              <button
                className="btn-danger"
                onClick={() => {
                  remove(modal.id)
                  setModal({ type: 'none' })
                }}
              >
                <Trash2 className="w-4 h-4" />
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
