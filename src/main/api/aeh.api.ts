import { ipcMain } from 'electron'
import { getAEHProcessManager } from '../infrastructure/aeh-server/server'
import { BackendClient } from '../infrastructure/python-server/client'
import { getBackendPort, getExternalApiToken } from '../infrastructure/python-server/server'

function aehClient(): BackendClient {
  const port = getAEHProcessManager().getPort()
  if (port === null) {
    throw new Error('AEH server is not running')
  }
  return new BackendClient(port)
}

// backend_url/backend_token are injected here since AEH's own subprocess never holds them.
function withBackendConnection(body: unknown): Record<string, unknown> {
  const port = getBackendPort()
  if (port === null) {
    throw new Error('CodeSpectra backend is not running')
  }
  return {
    ...(body as Record<string, unknown>),
    backend_url: `http://127.0.0.1:${port}`,
    backend_token: getExternalApiToken()
  }
}

export function registerAEHHandlers(): void {
  ipcMain.handle('aeh:start', async () => {
    const manager = getAEHProcessManager()
    const port = await manager.start()
    return port
  })

  ipcMain.handle('aeh:listRuns', () => aehClient().get('/api/runs'))

  ipcMain.handle('aeh:runDetail', (_e, runId: string) =>
    aehClient().get(`/api/runs/${runId}`)
  )

  ipcMain.handle('aeh:componentEvaluations', (_e, runId: string, componentId: string) =>
    aehClient().get(`/api/runs/${runId}/components/${componentId}`)
  )

  ipcMain.handle('aeh:traceDetail', (_e, traceId: string) =>
    aehClient().get(`/api/traces/${traceId}`)
  )

  ipcMain.handle('aeh:datasetCases', (_e, datasetId: string) =>
    aehClient().get(`/api/datasets/${datasetId}/cases`)
  )

  ipcMain.handle('aeh:providers', () => {
    // backend_url/backend_token must ride in query string as AEH's /api/providers is a GET request.
    const port = getBackendPort()
    if (port === null) {
      throw new Error('CodeSpectra backend is not running')
    }
    const qs = new URLSearchParams({
      backend_url: `http://127.0.0.1:${port}`,
      backend_token: getExternalApiToken()
    })
    return aehClient().get(`/api/providers?${qs.toString()}`)
  })

  ipcMain.handle('aeh:rerun', (_e, runId: string, body: unknown) =>
    aehClient().post(`/api/runs/${runId}/rerun`, body)
  )

  ipcMain.handle('aeh:startDiscovery', (_e, body: unknown) =>
    aehClient().post('/api/discovery/sessions', withBackendConnection(body))
  )

  ipcMain.handle('aeh:listDiscoverySessions', (_e, repoRef?: string, snapshotId?: string) => {
    // snapshot_id is the reliable filter, as repo_ref's value has been inconsistent across call sites.
    const params = new URLSearchParams()
    if (repoRef) params.set('repo_ref', repoRef)
    if (snapshotId) params.set('snapshot_id', snapshotId)
    const qs = params.toString()
    return aehClient().get(`/api/discovery/sessions${qs ? `?${qs}` : ''}`)
  })

  ipcMain.handle('aeh:getDiscoverySession', (_e, sessionId: string) =>
    aehClient().get(`/api/discovery/sessions/${sessionId}`)
  )

  ipcMain.handle('aeh:resumeDiscoverySession', (_e, sessionId: string, body: unknown) =>
    aehClient().post(`/api/discovery/sessions/${sessionId}/resume`, withBackendConnection(body ?? {}))
  )

  ipcMain.handle('aeh:listDiscoveryCandidates', (_e, sessionId: string) =>
    aehClient().get(`/api/discovery/sessions/${sessionId}/candidates`)
  )

  ipcMain.handle('aeh:updateDiscoveryCandidateVerdict', (_e, candidateId: string, verdict: string) =>
    aehClient().post(`/api/discovery/candidates/${candidateId}/verdict`, { verdict })
  )

  ipcMain.handle('aeh:updateDiscoveryCandidateExcludedFiles', (_e, candidateId: string, excludedFiles: string[]) =>
    aehClient().post(`/api/discovery/candidates/${candidateId}/excluded-files`, { excluded_files: excludedFiles })
  )

  ipcMain.handle('aeh:startExpansion', (_e, candidateId: string, body: unknown) =>
    aehClient().post(`/api/discovery/candidates/${candidateId}/expand`, withBackendConnection(body))
  )

  ipcMain.handle('aeh:getExpansionSession', (_e, sessionId: string) =>
    aehClient().get(`/api/discovery/expansion-sessions/${sessionId}`)
  )

  ipcMain.handle('aeh:listExpansionSessions', (_e, candidateId: string) =>
    aehClient().get(`/api/discovery/candidates/${candidateId}/expansion-sessions`)
  )

  ipcMain.handle('aeh:getExpansionMap', (_e, sessionId: string) =>
    aehClient().get(`/api/discovery/expansion-sessions/${sessionId}/map`)
  )

  ipcMain.handle('aeh:updateExpansionMap', (_e, sessionId: string, map: unknown) =>
    aehClient().put(`/api/discovery/expansion-sessions/${sessionId}/map`, map)
  )

  ipcMain.handle('aeh:generatePlan', (_e, sessionId: string, body: unknown) =>
    aehClient().post(`/api/discovery/expansion-sessions/${sessionId}/plan`, withBackendConnection(body))
  )

  ipcMain.handle('aeh:getPlan', (_e, sessionId: string) =>
    aehClient().get(`/api/discovery/expansion-sessions/${sessionId}/plan`)
  )

  ipcMain.handle('aeh:updatePlan', (_e, sessionId: string, body: unknown) =>
    aehClient().put(`/api/discovery/expansion-sessions/${sessionId}/plan`, body)
  )

  ipcMain.handle('aeh:advanceSession', (_e, sessionId: string, body: unknown) =>
    aehClient().post(`/api/discovery/sessions/${sessionId}/advance`, withBackendConnection(body))
  )
}
