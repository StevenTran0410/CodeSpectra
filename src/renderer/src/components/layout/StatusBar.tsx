import React, { useEffect, useState } from 'react'
import { Loader2 } from 'lucide-react'
import { useWorkspaceStore } from '../../store/workspace.store'
import { useJobStore } from '../../store/job.store'

export function StatusBar(): React.ReactElement {
  const { workspaces, activeWorkspaceId } = useWorkspaceStore()
  const activeJob = useJobStore(s => s.activeJob)
  const polling = useJobStore(s => s.polling)
  const [version, setVersion] = useState('')

  const activeWorkspace = workspaces.find((w) => w.id === activeWorkspaceId)

  useEffect(() => {
    window.api.app.getVersion().then(setVersion).catch(() => {})
  }, [])

  const statusText = polling
    ? 'Running...'
    : activeJob?.status === 'done'
      ? 'Completed'
      : activeJob?.status === 'failed'
        ? 'Failed'
        : 'No active run'

  const statusColor = polling || (activeJob?.status === 'done')
    ? 'text-emerald-400'
    : activeJob?.status === 'failed'
      ? 'text-rose-400'
      : 'text-gray-500'

  return (
    <footer className="h-7 flex items-center justify-between px-4 bg-surface border-t border-surface-border text-xs shrink-0 select-none">
      <span className="truncate">
        {activeWorkspace ? (
          <>
            <span className="text-gray-400">{activeWorkspace.name}</span>
            <span className="mx-1.5 opacity-40">·</span>
            <span className={`flex items-center gap-1.5 ${statusColor}`}>
              {polling && <Loader2 size={11} className="animate-spin" />}
              {statusText}
            </span>
          </>
        ) : (
          <span className="text-gray-500">No workspace selected</span>
        )}
      </span>
      <span className="shrink-0 text-gray-500">CodeSpectra {version}</span>
    </footer>
  )
}
