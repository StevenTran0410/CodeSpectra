import React, { Suspense, lazy } from 'react'
import { HashRouter, Routes, Route } from 'react-router-dom'
import { AppShell } from './components/layout/AppShell'
import { ErrorBoundary } from './components/ui/ErrorBoundary'
import { Skeleton } from './components/ui/LoadingSkeleton'

const Home = lazy(() => import('./routes/Home'))
const ProvidersSetup = lazy(() => import('./routes/ProvidersSetup'))
const CodeHostsSetup = lazy(() => import('./routes/CodeHostsSetup'))
const RepositoriesScreen = lazy(() => import('./routes/RepositoriesScreen'))
const AnalysisRunScreen = lazy(() => import('./routes/AnalysisRunScreen'))
const ReportViewerScreen = lazy(() => import('./routes/ReportViewerScreen'))
const SettingsScreen = lazy(() => import('./routes/SettingsScreen'))

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
      <HashRouter>
        <AppShell>
          <Suspense fallback={<PageFallback />}>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/providers" element={<ProvidersSetup />} />
              <Route path="/code-hosts" element={<CodeHostsSetup />} />
              <Route path="/repositories" element={<RepositoriesScreen />} />
              <Route path="/analysis" element={<AnalysisRunScreen />} />
              <Route path="/reports" element={<ReportViewerScreen />} />
              <Route path="/settings" element={<SettingsScreen />} />
            </Routes>
          </Suspense>
        </AppShell>
      </HashRouter>
    </ErrorBoundary>
  )
}
