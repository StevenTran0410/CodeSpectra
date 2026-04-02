import type { Workspace } from '../../../main/workspace/workspace.types'

declare global {
  interface Window {
    api: {
      workspace: {
        list: () => Promise<Workspace[]>
        create: (name: string) => Promise<Workspace>
        rename: (id: string, name: string) => Promise<Workspace>
        delete: (id: string) => Promise<void>
      }
      app: {
        getVersion: () => Promise<string>
        getUserDataPath: () => Promise<string>
        getLogsPath: () => Promise<string>
      }
    }
  }
}

export type { Workspace }
