import path from 'path'
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

  ipcMain.handle(
    'folder:branches',
    (_event, id: string): Promise<string[]> => client.get(`/api/local-repo/${id}/branches`)
  )

  ipcMain.handle(
    'folder:setBranch',
    (_event, id: string, branch: string) =>
      client.post(`/api/local-repo/${id}/branch`, { branch })
  )

  /**
   * Clone a remote git URL.
   * Destination is auto-resolved to ~/CodeSpectra/repos/<repo-name>.
   * Returns the registered LocalRepo on success.
   */
  ipcMain.handle('folder:cloneFromUrl', async (_event, url: string) => {
    const repoName = url.split('/').pop()?.replace(/\.git$/, '') ?? 'repo'
    const destPath = path.join(app.getPath('home'), 'CodeSpectra', 'repos', repoName)
    return client.post('/api/local-repo/clone', { url, dest_path: destPath })
  })

  // ── Git / SSH settings ────────────────────────────────────────────────────
  ipcMain.handle('git:getConfig', () => client.get('/api/app/git-config'))

  ipcMain.handle('git:setConfig', (_event, sshKeyPath: string | null) =>
    client.put('/api/app/git-config', { ssh_key_path: sshKeyPath })
  )

  ipcMain.handle('git:pickSshKey', async (): Promise<string | null> => {
    const result = await dialog.showOpenDialog({
      title: 'Select SSH Private Key',
      defaultPath: path.join(app.getPath('home'), '.ssh'),
      properties: ['openFile'],
      buttonLabel: 'Use This Key',
      filters: [
        { name: 'SSH Keys', extensions: ['*'] },
      ],
    })
    if (result.canceled || result.filePaths.length === 0) return null
    return result.filePaths[0]
  })
}
