import { ipcMain } from 'electron'
import type { BackendClient } from '../infrastructure/python-server/client'

export interface LocalEmbeddingStatus {
  enabled: boolean
  gpu_available: boolean
  vram_gb: number | null
  model_id: string
}

export function registerLocalEmbeddingHandlers(client: BackendClient): void {
  ipcMain.handle(
    'localEmbedding:status',
    (): Promise<LocalEmbeddingStatus> => client.get('/api/local-embedding/status')
  )

  ipcMain.handle(
    'localEmbedding:setEnabled',
    (_event, enabled: boolean): Promise<LocalEmbeddingStatus> =>
      client.post('/api/local-embedding/status', { enabled })
  )
}
