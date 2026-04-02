import { ipcMain } from 'electron'
import type { BackendClient } from '../infrastructure/python-server/client'
import type { ProviderConfig, CreateProviderRequest, UpdateProviderRequest } from './types'

export function registerProviderHandlers(client: BackendClient): void {
  ipcMain.handle('provider:list', (): Promise<ProviderConfig[]> =>
    client.get('/api/provider/')
  )

  ipcMain.handle(
    'provider:create',
    (_event, req: CreateProviderRequest): Promise<ProviderConfig> =>
      client.post('/api/provider/', req)
  )

  ipcMain.handle(
    'provider:update',
    (_event, id: string, req: UpdateProviderRequest): Promise<ProviderConfig> =>
      client.put(`/api/provider/${id}`, req)
  )

  ipcMain.handle('provider:delete', (_event, id: string): Promise<void> =>
    client.del(`/api/provider/${id}`)
  )

  ipcMain.handle(
    'provider:test',
    (_event, id: string): Promise<{ ok: boolean; message: string; warning?: string }> =>
      client.post(`/api/provider/${id}/test`, {})
  )

  ipcMain.handle(
    'provider:models',
    (_event, id: string): Promise<{ models: string[] }> =>
      client.get(`/api/provider/${id}/models`)
  )
}
