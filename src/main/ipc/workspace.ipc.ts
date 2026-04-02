import { ipcMain } from 'electron'
import type { WorkspaceService } from '../workspace/workspace.service'
import { logger } from '../logger'

export function registerWorkspaceHandlers(service: WorkspaceService): void {
  ipcMain.handle('workspace:list', () => {
    return service.list()
  })

  ipcMain.handle('workspace:create', (_event, name: string) => {
    return service.create(name)
  })

  ipcMain.handle('workspace:rename', (_event, id: string, name: string) => {
    return service.rename(id, name)
  })

  ipcMain.handle('workspace:delete', (_event, id: string) => {
    service.delete(id)
  })

  logger.debug('Workspace IPC handlers registered')
}
