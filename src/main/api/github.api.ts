import { ipcMain, shell } from 'electron'
import type { BackendClient } from '../infrastructure/python-server/client'
import type {
  DeviceFlowStart,
  DeviceFlowPollResult,
  GitHubAccount,
  GitHubRepoListResponse,
} from './types'

export function registerGitHubHandlers(client: BackendClient): void {
  /** Start GitHub OAuth device flow — returns user_code and verification_uri. */
  ipcMain.handle(
    'github:startDeviceFlow',
    (): Promise<DeviceFlowStart> => client.post('/api/github/device-flow/start', {})
  )

  /** Poll once for the result of the device flow. */
  ipcMain.handle(
    'github:pollDeviceFlow',
    (_event, deviceCode: string): Promise<DeviceFlowPollResult> =>
      client.post('/api/github/device-flow/poll', { device_code: deviceCode })
  )

  /** Open the GitHub verification URL in the system browser. */
  ipcMain.handle(
    'github:openBrowser',
    (_event, url: string): Promise<void> => shell.openExternal(url)
  )

  /** Return the connected GitHub account (null if not connected). */
  ipcMain.handle(
    'github:getAccount',
    (): Promise<GitHubAccount | null> => client.get('/api/github/account')
  )

  /** Disconnect and delete the stored GitHub token. */
  ipcMain.handle(
    'github:disconnect',
    (): Promise<void> => client.del('/api/github/account')
  )

  /** List repositories for the connected account, with optional search + pagination. */
  ipcMain.handle(
    'github:listRepos',
    (_event, query?: string, page = 1): Promise<GitHubRepoListResponse> => {
      const params = new URLSearchParams({ page: String(page) })
      if (query) params.set('query', query)
      return client.get(`/api/github/repos?${params.toString()}`)
    }
  )
}
