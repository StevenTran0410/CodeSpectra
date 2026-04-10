import React, { useEffect, useState, type ReactNode } from 'react'
import { Sidebar } from './Sidebar'
import { StatusBar } from './StatusBar'
import { ErrorBoundary } from '../ui/ErrorBoundary'
import { useWorkspaceStore } from '../../store/workspace.store'
import { OnboardingWizard } from '../onboarding/OnboardingWizard'

interface Props {
  children: ReactNode
}

export function AppShell({ children }: Props): React.ReactElement {
  const load = useWorkspaceStore((s) => s.load)
  const isLoading = useWorkspaceStore((s) => s.isLoading)
  const workspaces = useWorkspaceStore((s) => s.workspaces)
  const error = useWorkspaceStore((s) => s.error)
  // null = not yet determined (still loading); true = wizard active; false = dismissed
  const [wizardActive, setWizardActive] = useState<boolean | null>(null)

  useEffect(() => {
    load()
  }, [load])

  // Only decide once, after the first load completes
  useEffect(() => {
    if (!isLoading && wizardActive === null) {
      setWizardActive(workspaces.length === 0)
    }
  }, [isLoading, wizardActive])

  if (wizardActive) {
    return <OnboardingWizard onDone={() => setWizardActive(false)} />
  }

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
