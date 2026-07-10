import { ipcMain } from 'electron'
import type { BackendClient } from '../infrastructure/python-server/client'

export interface LocalEmbeddingStatus {
  enabled: boolean
  gpu_available: boolean
  vram_gb: number | null
  model_id: string
  weights_ready: boolean
  weights_dir: string
  download_url: string
  download_status: 'idle' | 'downloading' | 'done' | 'failed'
  download_error: string | null
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

  ipcMain.handle(
    'localEmbedding:download',
    (): Promise<LocalEmbeddingStatus> => client.post('/api/local-embedding/download', {})
  )
}
