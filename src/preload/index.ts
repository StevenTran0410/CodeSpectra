import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'
import type {
  Workspace,
  ProviderConfig,
  CreateProviderRequest,
  UpdateProviderRequest
} from '../main/api/types'

const api = {
  workspace: {
    list: (): Promise<Workspace[]> => ipcRenderer.invoke('workspace:list'),
    create: (name: string): Promise<Workspace> => ipcRenderer.invoke('workspace:create', name),
    rename: (id: string, name: string): Promise<Workspace> =>
      ipcRenderer.invoke('workspace:rename', id, name),
    delete: (id: string): Promise<void> => ipcRenderer.invoke('workspace:delete', id)
  },
  provider: {
    list: (): Promise<ProviderConfig[]> => ipcRenderer.invoke('provider:list'),
    create: (req: CreateProviderRequest): Promise<ProviderConfig> =>
      ipcRenderer.invoke('provider:create', req),
    update: (id: string, req: UpdateProviderRequest): Promise<ProviderConfig> =>
      ipcRenderer.invoke('provider:update', id, req),
    delete: (id: string): Promise<void> => ipcRenderer.invoke('provider:delete', id),
    test: (id: string): Promise<{ ok: boolean; message: string; warning?: string }> =>
      ipcRenderer.invoke('provider:test', id),
    models: (id: string): Promise<{ models: string[] }> =>
      ipcRenderer.invoke('provider:models', id)
  },
  consent: {
    checkCloud: (): Promise<{ given: boolean }> => ipcRenderer.invoke('consent:cloud:check'),
    giveCloud: (given: boolean): Promise<{ given: boolean }> =>
      ipcRenderer.invoke('consent:cloud:give', given)
  },
  folder: {
    pick: (): Promise<string | null> => ipcRenderer.invoke('folder:pick'),
    validate: (path: string) => ipcRenderer.invoke('folder:validate', path),
    list: () => ipcRenderer.invoke('folder:list'),
    add: (path: string) => ipcRenderer.invoke('folder:add', path),
    remove: (id: string): Promise<void> => ipcRenderer.invoke('folder:remove', id),
    revalidate: (id: string) => ipcRenderer.invoke('folder:revalidate', id)
  },
  app: {
    getVersion: (): Promise<string> => ipcRenderer.invoke('app:get-version'),
    getUserDataPath: (): Promise<string> => ipcRenderer.invoke('app:get-user-data-path'),
    getLogsPath: (): Promise<string> => ipcRenderer.invoke('app:get-logs-path')
  }
}

if (process.contextIsolated) {
  contextBridge.exposeInMainWorld('electron', electronAPI)
  contextBridge.exposeInMainWorld('api', api)
} else {
  // @ts-expect-error fallback
  window.electron = electronAPI
  // @ts-expect-error fallback
  window.api = api
}
