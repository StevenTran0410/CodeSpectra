import { ipcMain, app } from 'electron'
import log from 'electron-log'
import { HttpClient } from '../http-client'

export function registerAppHandlers(client: HttpClient): void {
  ipcMain.handle('app:get-version', () => app.getVersion())
  ipcMain.handle('app:get-user-data-path', () => app.getPath('userData'))
  ipcMain.handle('app:get-logs-path', () => log.transports.file.getFile().path)
  ipcMain.handle('app:get-diagnostics', () => client.get('/api/app/diagnostics'))
}
