import { ipcMain, WebContentsView, BrowserWindow } from 'electron'
import { getAEHProcessManager } from '../infrastructure/aeh-server/server'

let aehView: WebContentsView | null = null
let lastLoadedPort: number | null = null

export function registerAEHHandlers(): void {
  ipcMain.handle('aeh:start', async () => {
    const manager = getAEHProcessManager()
    const port = await manager.start()
    return port
  })

  ipcMain.handle(
    'aeh:show-view',
    async (_event, bounds: { x: number; y: number; width: number; height: number }) => {
      const manager = getAEHProcessManager()
      const port = manager.getPort()
      if (port === null) {
        throw new Error('AEH server is not running')
      }

      const win = BrowserWindow.getAllWindows()[0]
      if (!win) return

      if (!aehView) {
        aehView = new WebContentsView({
          webPreferences: {
            webSecurity: true,
            sandbox: false
          }
        })
        aehView.webContents.loadURL(`http://127.0.0.1:${port}/`)
        lastLoadedPort = port
      } else if (lastLoadedPort !== port) {
        aehView.webContents.loadURL(`http://127.0.0.1:${port}/`)
        lastLoadedPort = port
      }

      // Attach it if not already attached
      const children = win.contentView.children
      if (!children.includes(aehView)) {
        win.contentView.addChildView(aehView)
      }

      aehView.setBounds(bounds)
    }
  )

  ipcMain.handle('aeh:hide-view', async () => {
    const win = BrowserWindow.getAllWindows()[0]
    if (win && aehView) {
      win.contentView.removeChildView(aehView)
    }
  })

  ipcMain.handle(
    'aeh:resize-view',
    async (_event, bounds: { x: number; y: number; width: number; height: number }) => {
      if (aehView) {
        aehView.setBounds(bounds)
      }
    }
  )
}

export function stopAEHView(): void {
  if (aehView) {
    aehView = null
  }
}
