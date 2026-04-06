import { app, BrowserWindow, dialog } from 'electron'
import { electronApp, optimizer } from '@electron-toolkit/utils'
import { createMainWindow } from './window'
import { startPythonServer, stopPythonServer } from './infrastructure/python-server/server'
import { registerWorkspaceHandlers } from './api/workspace.api'
import { registerProviderHandlers } from './api/provider.api'
import { registerConsentHandlers } from './api/consent.api'
import { registerFolderHandlers } from './api/folder.api'
import { registerJobHandlers } from './api/job.api'
import { registerAppHandlers } from './api/app.api'
import { logger } from './shared/logger'

app.whenReady().then(async () => {
  electronApp.setAppUserModelId('com.codespectra.app')

  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  try {
    logger.info(`CodeSpectra ${app.getVersion()} — starting Python backend...`)
    const client = await startPythonServer()

    registerAppHandlers(client)
    registerWorkspaceHandlers(client)
    registerProviderHandlers(client)
    registerConsentHandlers(client)
    registerFolderHandlers(client)
    registerJobHandlers(client)

    createMainWindow()
    logger.info('Startup complete')
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    logger.error('Fatal startup error:', err)
    await dialog.showErrorBox(
      'CodeSpectra — Startup Failed',
      `Could not start the analysis engine.\n\n${message}\n\nMake sure Python 3.11+ is installed and try again.`
    )
    app.quit()
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createMainWindow()
  })
})

app.on('window-all-closed', () => {
  stopPythonServer()
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', () => {
  stopPythonServer()
})
