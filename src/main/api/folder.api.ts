import { app, dialog, ipcMain } from 'electron'
import type { BackendClient } from '../infrastructure/python-server/client'

export function registerFolderHandlers(client: BackendClient): void {
  /** Open native folder-picker dialog and return the selected path (or null if cancelled). */
  ipcMain.handle('folder:pick', async (): Promise<string | null> => {
    const result = await dialog.showOpenDialog({
      title: 'Select Repository Folder',
      defaultPath: app.getPath('home'),
      properties: ['openDirectory'],
      buttonLabel: 'Open Folder',
    })
    if (result.canceled || result.filePaths.length === 0) return null
    return result.filePaths[0]
  })

  ipcMain.handle(
    'folder:validate',
    (_event, path: string) => client.post('/api/local-repo/validate', { path })
  )

  ipcMain.handle(
    'folder:list',
    () => client.get('/api/local-repo/')
  )

  ipcMain.handle(
    'folder:add',
    (_event, path: string) => client.post('/api/local-repo/', { path })
  )

  ipcMain.handle(
    'folder:remove',
    (_event, id: string) => client.del(`/api/local-repo/${id}`)
  )

  ipcMain.handle(
    'folder:revalidate',
    (_event, id: string) => client.post(`/api/local-repo/${id}/revalidate`, {})
  )
}
