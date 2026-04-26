import { ipcMain } from 'electron'
import type { BackendClient } from '../infrastructure/python-server/client'

export function registerImpactHandlers(client: BackendClient): void {
  ipcMain.handle(
    'impact:blastRadius',
    (_event, body: {
      snapshot_id: string
      changed_files: string[]
      report_id?: string | null
      max_hops?: number
      include_call_chains?: boolean
    }) => client.post('/api/impact/blast-radius', body)
  )

  ipcMain.handle(
    'impact:plan',
    (_event, body: {
      snapshot_id: string
      task_description: string
      report_id: string
      provider_id?: string | null
      model_id?: string | null
    }) => client.post('/api/impact/plan', body)
  )
}
