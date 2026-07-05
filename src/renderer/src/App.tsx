import React, { Suspense, lazy } from 'react'
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AppShell } from './components/layout/AppShell'
import { ErrorBoundary } from './components/ui/ErrorBoundary'
import { Skeleton } from './components/ui/LoadingSkeleton'
import { ToastContainer } from './components/ui'

const Home = lazy(() => import('./screens/home'))
const ProvidersSetup = lazy(() => import('./screens/providers'))
const CodeHostsSetup = lazy(() => import('./screens/code-hosts'))
const RepositoriesScreen = lazy(() => import('./screens/repositories'))
const SnapshotViewerScreen = lazy(() => import('./screens/snapshot-viewer'))
const IndexOverviewScreen = lazy(() => import('./screens/index-overview'))
const AnalysisRunScreen = lazy(() => import('./screens/analysis'))
const ReportViewerScreen = lazy(() => import('./screens/reports'))
const AskScreen = lazy(() => import('./screens/ask'))
const GraphScreen = lazy(() => import('./screens/graph'))
const SettingsScreen = lazy(() => import('./screens/settings'))
const AEHReportsScreen = lazy(() => import('./screens/aeh-reports'))
const AEHAnalysisScreen = lazy(() => import('./screens/aeh/analysis'))

function PageFallback(): React.ReactElement {
  return (
    <div className="p-6 space-y-4">
      <Skeleton className="h-6 w-40" />
      <Skeleton className="h-4 w-64" />
      <Skeleton className="h-32 w-full" />
    </div>
  )
}

export default function App(): React.ReactElement {
  return (
    <ErrorBoundary>
      <ToastContainer />
      <HashRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <AppShell>
          <Suspense fallback={<PageFallback />}>
            <Routes>
              <Route path="/" element={<Home />} />
              {/* global, mode-independent */}
              <Route path="/providers" element={<ProvidersSetup />} />
              <Route path="/code-hosts" element={<CodeHostsSetup />} />
              <Route path="/snapshot-viewer" element={<SnapshotViewerScreen />} />
              <Route path="/index-overview" element={<IndexOverviewScreen />} />
              <Route path="/graph" element={<GraphScreen />} />
              <Route path="/settings" element={<SettingsScreen />} />

              {/* Code Analysis mode */}
              <Route path="/ca/repositories" element={<RepositoriesScreen />} />
              <Route path="/ca/analysis" element={<AnalysisRunScreen />} />
              <Route path="/ca/reports" element={<ReportViewerScreen />} />
              <Route path="/ca/ask" element={<AskScreen />} />

              {/* AEH mode */}
              <Route path="/aeh/repositories" element={<RepositoriesScreen />} />
              <Route path="/aeh/analysis" element={<AEHAnalysisScreen />} />
              <Route path="/aeh/reports/*" element={<AEHReportsScreen />} />
              <Route path="/aeh/ask" element={<AskScreen />} />

              {/* legacy redirects */}
              <Route path="/repositories" element={<Navigate to="/ca/repositories" replace />} />
              <Route path="/analysis" element={<Navigate to="/ca/analysis" replace />} />
              <Route path="/reports" element={<Navigate to="/ca/reports" replace />} />
              <Route path="/ask" element={<Navigate to="/ca/ask" replace />} />
              <Route path="/aeh" element={<Navigate to="/aeh/reports" replace />} />
            </Routes>
          </Suspense>
        </AppShell>
      </HashRouter>
    </ErrorBoundary>
  )
}
