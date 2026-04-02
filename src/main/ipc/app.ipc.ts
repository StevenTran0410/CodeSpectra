import { ipcMain, app } from 'electron'
import { logger } from '../logger'

export function registerAppHandlers(): void {
  ipcMain.handle('app:get-version', () => app.getVersion())

  ipcMain.handle('app:get-user-data-path', () => app.getPath('userData'))

  ipcMain.handle('app:get-logs-path', () => app.getPath('logs'))

  logger.debug('App IPC handlers registered')
}
