import React, { useEffect } from 'react'
import { Routes, Route } from 'react-router-dom'
import { useAEHStore } from '../../store/aeh.store'
import RunsListScreen from './RunsListScreen'
import RunDetailScreen from './RunDetailScreen'
import ComponentDetailScreen from './ComponentDetailScreen'
import TraceViewerScreen from './TraceViewerScreen'
import RunCompareScreen from './RunCompareScreen'

export default function AEHReportsScreen(): React.ReactElement {
  const { startAEH, isLoading, error } = useAEHStore()

  useEffect(() => {
    startAEH().catch(() => {})
  }, [startAEH])

  if (isLoading) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center min-h-[calc(100vh-64px)] text-slate-100 bg-[#090d16] gap-4">
        <div className="h-10 w-10 border-t-2 border-indigo-500 rounded-full animate-spin"></div>
        <p className="text-xs text-gray-400 font-medium">Starting AEH Dashboard backend...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center min-h-[calc(100vh-64px)] text-slate-100 bg-[#090d16] p-6 text-center gap-3">
        <p className="text-sm font-semibold text-red-400">Failed to launch evaluation dashboard</p>
        <p className="text-xs text-gray-500 max-w-sm">{error}</p>
      </div>
    )
  }

  return (
    <Routes>
      <Route path="/" element={<RunsListScreen />} />
      <Route path="runs/:runId" element={<RunDetailScreen />} />
      <Route path="runs/:runId/components/:componentId" element={<ComponentDetailScreen />} />
      <Route path="runs/:runId/traces/:traceId" element={<TraceViewerScreen />} />
      <Route path="compare/:runIdA/:runIdB" element={<RunCompareScreen />} />
    </Routes>
  )
}
