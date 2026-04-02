import React from 'react'
import { GitBranch } from 'lucide-react'
import { EmptyState } from '../../components/ui/EmptyState'

export default function CodeHostsSetup(): React.ReactElement {
  return (
    <>
      <div className="screen-header">
        <h1 className="screen-title">Code Hosts</h1>
        <p className="screen-subtitle">Connect GitHub, Bitbucket, or open a local folder</p>
      </div>
      <div className="h-[calc(100vh-10rem)]">
        <EmptyState
          icon={<GitBranch className="w-7 h-7" />}
          title="No code hosts connected"
          description="Connect a code host to discover and clone repositories, or open a local folder directly."
          action={
            <span className="text-xs text-gray-500 bg-surface-overlay border border-surface-border px-3 py-1.5 rounded-md">
              Coming in RPA-023 / RPA-025
            </span>
          }
        />
      </div>
    </>
  )
}
