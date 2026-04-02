import React from 'react'
import { FlaskConical } from 'lucide-react'
import { EmptyState } from '../components/ui/EmptyState'

export default function AnalysisRunScreen(): React.ReactElement {
  return (
    <>
      <div className="screen-header">
        <h1 className="screen-title">Analysis</h1>
        <p className="screen-subtitle">Run analysis on a repository and track progress</p>
      </div>
      <div className="h-[calc(100vh-10rem)]">
        <EmptyState
          icon={<FlaskConical className="w-7 h-7" />}
          title="No analysis runs yet"
          description="Select a repository and start an analysis run. You can choose between Quick Scan and Full Scan."
          action={
            <span className="text-xs text-gray-500 bg-surface-overlay border border-surface-border px-3 py-1.5 rounded-md">
              Coming in RPA-035
            </span>
          }
        />
      </div>
    </>
  )
}
