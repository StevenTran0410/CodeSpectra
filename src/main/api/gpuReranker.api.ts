import { ipcMain } from 'electron'
import type { BackendClient } from '../infrastructure/python-server/client'

export interface GpuRerankerStatus {
  enabled: boolean
  gpu_available: boolean
  vram_gb: number | null
}

export function registerGpuRerankerHandlers(client: BackendClient): void {
  ipcMain.handle(
    'gpuReranker:status',
    (): Promise<GpuRerankerStatus> => client.get('/api/gpu-reranker/status')
  )

  ipcMain.handle(
    'gpuReranker:setEnabled',
    (_event, enabled: boolean): Promise<GpuRerankerStatus> =>
      client.post('/api/gpu-reranker/status', { enabled })
  )
}
