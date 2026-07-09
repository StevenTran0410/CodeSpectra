import { ipcMain, app } from 'electron'
import log from 'electron-log'
import { BackendClient } from '../infrastructure/python-server/client'
import { getPythonProcessManager } from '../infrastructure/python-server/server'

export function registerAppHandlers(client: BackendClient): void {
  ipcMain.handle('app:get-version', () => app.getVersion())
  ipcMain.handle('app:get-user-data-path', () => app.getPath('userData'))
  ipcMain.handle('app:get-logs-path', () => log.transports.file.getFile().path)
  ipcMain.handle('app:get-diagnostics', () => client.get('/api/app/diagnostics'))

  // Allow manual backend restart retry
  ipcMain.handle('app:retry-backend', async () => {
    const manager = getPythonProcessManager()
    await manager.start().catch((err) => {
      throw new Error(`Backend restart failed: ${err.message}`)
    })
  })
}
