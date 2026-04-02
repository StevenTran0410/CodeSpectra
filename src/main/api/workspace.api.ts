import { ipcMain } from 'electron'
import type { BackendClient } from '../infrastructure/python-server/client'
import type { Workspace } from './types'

export function registerWorkspaceHandlers(client: BackendClient): void {
  ipcMain.handle('workspace:list', (): Promise<Workspace[]> =>
    client.get('/api/workspace/')
  )

  ipcMain.handle('workspace:create', (_event, name: string): Promise<Workspace> =>
    client.post('/api/workspace/', { name })
  )

  ipcMain.handle('workspace:rename', (_event, id: string, name: string): Promise<Workspace> =>
    client.put(`/api/workspace/${id}`, { name })
  )

  ipcMain.handle('workspace:delete', (_event, id: string): Promise<void> =>
    client.del(`/api/workspace/${id}`)
  )
}
