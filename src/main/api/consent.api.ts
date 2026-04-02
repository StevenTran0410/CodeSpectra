import { ipcMain } from 'electron'
import type { BackendClient } from '../infrastructure/python-server/client'

export function registerConsentHandlers(client: BackendClient): void {
  ipcMain.handle(
    'consent:cloud:check',
    (): Promise<{ given: boolean }> => client.get('/api/consent/cloud')
  )

  ipcMain.handle(
    'consent:cloud:give',
    (_event, given: boolean): Promise<{ given: boolean }> =>
      client.post('/api/consent/cloud', { given })
  )
}
