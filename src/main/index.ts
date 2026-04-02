import { app, BrowserWindow } from 'electron'
import { electronApp, optimizer } from '@electron-toolkit/utils'
import { createMainWindow } from './window'
import { initDb, closeDb } from './db/database'
import { WorkspaceService } from './workspace/workspace.service'
import { registerWorkspaceHandlers } from './ipc/workspace.ipc'
import { registerAppHandlers } from './ipc/app.ipc'
import { logger } from './logger'

app.whenReady().then(() => {
  electronApp.setAppUserModelId('com.codespectra.app')

  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  try {
    const db = initDb()
    const workspaceService = new WorkspaceService(db)

    registerAppHandlers()
    registerWorkspaceHandlers(workspaceService)

    createMainWindow()

    logger.info(`CodeSpectra ${app.getVersion()} started`)
  } catch (err) {
    logger.error('Fatal error during startup:', err)
    app.quit()
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createMainWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    closeDb()
    app.quit()
  }
})

app.on('before-quit', () => {
  closeDb()
})
