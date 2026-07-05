import { app, BrowserWindow, dialog } from 'electron'
import { electronApp, is, optimizer } from '@electron-toolkit/utils'

// Suppress Electron security warnings in dev (unsafe-eval is required by Vite HMR)
if (is.dev) process.env.ELECTRON_DISABLE_SECURITY_WARNINGS = 'true'

// Silence harmless Chromium-internal DevTools noise ("Autofill.enable wasn't
// found" / "Autofill.setAddresses wasn't found"). This is a known DevTools
// frontend/backend version mismatch inside Electron's bundled Chromium (the
// inspector frontend calls a CDP method this Chromium build doesn't
// implement) — it cannot be intercepted via any JS API since it's native
// Chromium stderr logging, not our app's console. Raising the native log
// threshold to FATAL-only hides it without touching our own logger (a
// separate channel via shared/logger.ts), so real app errors are unaffected.
app.commandLine.appendSwitch('log-level', '3')
import { createMainWindow } from './window'
import { startPythonServer, stopPythonServer } from './infrastructure/python-server/server'
import { registerWorkspaceHandlers } from './api/workspace.api'
import { registerProviderHandlers } from './api/provider.api'
import { registerConsentHandlers } from './api/consent.api'
import { registerGpuRerankerHandlers } from './api/gpuReranker.api'
import { registerFolderHandlers } from './api/folder.api'
import { registerJobHandlers } from './api/job.api'
import { registerAppHandlers } from './api/app.api'
import { registerQAHandlers } from './api/qa.api'
import { registerImpactHandlers } from './api/impact.api'
import { registerAEHHandlers } from './api/aeh.api'
import { getAEHProcessManager } from './infrastructure/aeh-server/server'
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
    registerGpuRerankerHandlers(client)
    registerFolderHandlers(client)
    registerJobHandlers(client)
    registerQAHandlers(client)
    registerImpactHandlers(client)
    registerAEHHandlers()

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
  getAEHProcessManager().stop()
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', () => {
  stopPythonServer()
  getAEHProcessManager().stop()
})
