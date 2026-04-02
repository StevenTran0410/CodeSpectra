import { ipcMain, app } from 'electron'
import log from 'electron-log'

export function registerAppHandlers(): void {
  ipcMain.handle('app:get-version', () => app.getVersion())
  ipcMain.handle('app:get-user-data-path', () => app.getPath('userData'))
  ipcMain.handle('app:get-logs-path', () => log.transports.file.getFile().path)
}
