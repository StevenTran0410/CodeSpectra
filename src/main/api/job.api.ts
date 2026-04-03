import { ipcMain } from 'electron'
import type { BackendClient } from '../infrastructure/python-server/client'

export function registerJobHandlers(client: BackendClient): void {
  ipcMain.handle('job:get', (_e, id: string) => client.get(`/api/job/${id}`))

  ipcMain.handle('job:cancel', (_e, id: string) =>
    client.post(`/api/job/${id}/cancel`, {})
  )

  ipcMain.handle('job:listForRepo', (_e, repoId: string) =>
    client.get(`/api/job/repo/${repoId}`)
  )

  ipcMain.handle('job:listRecent', () => client.get('/api/job/'))
}
