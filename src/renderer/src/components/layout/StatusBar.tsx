import React, { useEffect, useState } from 'react'
import { useWorkspaceStore } from '../../store/workspace.store'

export function StatusBar(): React.ReactElement {
  const { workspaces, activeWorkspaceId } = useWorkspaceStore()
  const [version, setVersion] = useState('')

  const activeWorkspace = workspaces.find((w) => w.id === activeWorkspaceId)

  useEffect(() => {
    window.api.app.getVersion().then(setVersion).catch(() => {})
  }, [])

  return (
    <footer className="h-7 flex items-center justify-between px-4 bg-surface border-t border-surface-border text-xs text-gray-500 shrink-0 select-none">
      <span className="truncate">
        {activeWorkspace ? (
          <>
            <span className="text-gray-400">{activeWorkspace.name}</span>
            <span className="mx-1.5 opacity-40">·</span>
            <span>No active run</span>
          </>
        ) : (
          'No workspace selected'
        )}
      </span>
      <span className="shrink-0">CodeSpectra {version}</span>
    </footer>
  )
}
