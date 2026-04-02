import React, { useEffect, type ReactNode } from 'react'
import { Sidebar } from './Sidebar'
import { StatusBar } from './StatusBar'
import { ErrorBoundary } from '../ui/ErrorBoundary'
import { useWorkspaceStore } from '../../store/workspace.store'

interface Props {
  children: ReactNode
}

export function AppShell({ children }: Props): React.ReactElement {
  const load = useWorkspaceStore((s) => s.load)

  useEffect(() => {
    load()
  }, [load])

  return (
    <div className="flex h-screen overflow-hidden bg-surface text-gray-100">
      <Sidebar />
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <main className="flex-1 overflow-auto">
          <ErrorBoundary>{children}</ErrorBoundary>
        </main>
        <StatusBar />
      </div>
    </div>
  )
}
