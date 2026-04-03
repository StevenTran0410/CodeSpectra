import { BrowserWindow, Menu, session } from 'electron'
import { is } from '@electron-toolkit/utils'
import path from 'path'
import { logger } from './shared/logger'

export function createMainWindow(): BrowserWindow {
  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    show: false,
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    backgroundColor: '#0f1117',
    webPreferences: {
      preload: path.join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      webSecurity: true
    }
  })

  // CSP: strict in production, relaxed in dev (Vite HMR needs unsafe-inline + unsafe-eval)
  const csp = is.dev
    ? "default-src 'self'; " +
      "script-src 'self' 'unsafe-inline' 'unsafe-eval'; " +
      "style-src 'self' 'unsafe-inline'; " +
      "img-src 'self' data: blob:; " +
      "font-src 'self' data:; " +
      "connect-src 'self' ws://localhost:* http://localhost:* http://127.0.0.1:*; " +
      "frame-src 'none'"
    : "default-src 'self'; " +
      "script-src 'self'; " +
      "style-src 'self' 'unsafe-inline'; " +
      "img-src 'self' data: blob:; " +
      "font-src 'self' data:; " +
      "connect-src 'self' http://localhost:* http://127.0.0.1:*; " +
      "frame-src 'none'"

  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [csp]
      }
    })
  })

  win.on('ready-to-show', () => {
    win.show()
    if (is.dev) win.webContents.openDevTools({ mode: 'right' })
  })

  win.webContents.setWindowOpenHandler(() => {
    return { action: 'deny' }
  })

  // Right-click context menu for text inputs (copy / paste / cut / select all)
  win.webContents.on('context-menu', (_event, params) => {
    if (!params.isEditable && !params.selectionText) return
    const menu = Menu.buildFromTemplate([
      { role: 'cut',       enabled: params.editFlags.canCut,       label: 'Cut' },
      { role: 'copy',      enabled: params.editFlags.canCopy,      label: 'Copy' },
      { role: 'paste',     enabled: params.editFlags.canPaste,     label: 'Paste' },
      { type: 'separator' },
      { role: 'selectAll', enabled: params.editFlags.canSelectAll, label: 'Select All' },
    ])
    menu.popup({ window: win })
  })

  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    win.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    win.loadFile(path.join(__dirname, '../renderer/index.html'))
  }

  logger.info('Main window created')
  return win
}
