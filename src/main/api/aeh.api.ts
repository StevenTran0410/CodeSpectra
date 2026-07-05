import { ipcMain } from 'electron'
import { getAEHProcessManager } from '../infrastructure/aeh-server/server'
import { BackendClient } from '../infrastructure/python-server/client'

function aehClient(): BackendClient {
  const port = getAEHProcessManager().getPort()
  if (port === null) {
    throw new Error('AEH server is not running')
  }
  return new BackendClient(port)
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

  ipcMain.handle('aeh:providers', () =>
    aehClient().get('/api/providers')
  )

  ipcMain.handle('aeh:rerun', (_e, runId: string, body: unknown) =>
    aehClient().post(`/api/runs/${runId}/rerun`, body)
  )

  ipcMain.handle('aeh:startDiscovery', (_e, body: unknown) =>
    aehClient().post('/api/discovery/sessions', body)
  )

  ipcMain.handle('aeh:listDiscoverySessions', (_e, repoRef?: string) => {
    const url = repoRef ? `/api/discovery/sessions?repo_ref=${encodeURIComponent(repoRef)}` : '/api/discovery/sessions'
    return aehClient().get(url)
  })

  ipcMain.handle('aeh:getDiscoverySession', (_e, sessionId: string) =>
    aehClient().get(`/api/discovery/sessions/${sessionId}`)
  )

  ipcMain.handle('aeh:listDiscoveryCandidates', (_e, sessionId: string) =>
    aehClient().get(`/api/discovery/sessions/${sessionId}/candidates`)
  )

  ipcMain.handle('aeh:updateDiscoveryCandidateVerdict', (_e, candidateId: string, verdict: string) =>
    aehClient().post(`/api/discovery/candidates/${candidateId}/verdict`, { verdict })
  )
}
