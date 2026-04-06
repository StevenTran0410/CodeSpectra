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

  // Block all new windows (popups, target="_blank", window.open)
  win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))

  // Block navigation away from the app bundle — prevents renderer-initiated
  // open-redirect attacks (e.g. a malicious repo with an HTML file that tries
  // to redirect to a remote URL).
  const isLocalUrl = (url: string): boolean => {
    try {
      const { protocol, hostname } = new URL(url)
      return (
        protocol === 'file:' ||
        ((protocol === 'http:' || protocol === 'https:') &&
          (hostname === 'localhost' || hostname === '127.0.0.1'))
      )
    } catch {
      return false
    }
  }

  win.webContents.on('will-navigate', (event, url) => {
    if (!isLocalUrl(url)) {
      event.preventDefault()
      logger.warn(`Blocked navigation to external URL: ${url}`)
    }
  })

  win.webContents.on('will-redirect', (event, url) => {
    if (!isLocalUrl(url)) {
      event.preventDefault()
      logger.warn(`Blocked redirect to external URL: ${url}`)
    }
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
