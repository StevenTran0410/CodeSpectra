import React from 'react'
import { FolderOpen } from 'lucide-react'
import { EmptyState } from '../../components/ui/EmptyState'
import { useWorkspaceStore } from '../../store/workspace.store'

export default function RepositoriesScreen(): React.ReactElement {
  const activeWorkspaceId = useWorkspaceStore((s) => s.activeWorkspaceId)

  return (
    <>
      <div className="screen-header">
        <h1 className="screen-title">Repositories</h1>
        <p className="screen-subtitle">Manage repository connections and snapshots</p>
      </div>
      <div className="h-[calc(100vh-10rem)]">
        {!activeWorkspaceId ? (
          <EmptyState
            icon={<FolderOpen className="w-7 h-7" />}
            title="No workspace selected"
            description="Select or create a workspace first, then add repository connections."
          />
        ) : (
          <EmptyState
            icon={<FolderOpen className="w-7 h-7" />}
            title="No repositories in this workspace"
            description="Connect a code host and add repositories to get started."
            action={
              <span className="text-xs text-gray-500 bg-surface-overlay border border-surface-border px-3 py-1.5 rounded-md">
                Coming in RPA-030
              </span>
            }
          />
        )}
      </div>
    </>
  )
}
