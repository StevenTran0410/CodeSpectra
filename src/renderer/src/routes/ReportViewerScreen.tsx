import React from 'react'
import { FileText } from 'lucide-react'
import { EmptyState } from '../components/ui/EmptyState'

export default function ReportViewerScreen(): React.ReactElement {
  return (
    <>
      <div className="screen-header">
        <h1 className="screen-title">Reports</h1>
        <p className="screen-subtitle">View, compare, and export analysis reports</p>
      </div>
      <div className="h-[calc(100vh-10rem)]">
        <EmptyState
          icon={<FileText className="w-7 h-7" />}
          title="No reports yet"
          description="Reports appear here after a successful analysis run. Each section has evidence and confidence scores."
          action={
            <span className="text-xs text-gray-500 bg-surface-overlay border border-surface-border px-3 py-1.5 rounded-md">
              Coming in RPA-043
            </span>
          }
        />
      </div>
    </>
  )
}
