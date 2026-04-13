import { ipcMain } from 'electron'
import type { BackendClient } from '../infrastructure/python-server/client'

export function registerQAHandlers(client: BackendClient): void {
  ipcMain.handle(
    'qa:ask',
    (_event, body: {
      snapshot_id: string
      question: string
      provider_id: string
      model_id: string
      report_id?: string
      include_debug?: boolean
    }) => client.post('/api/qa/ask', body)
  )

  ipcMain.handle(
    'qa:deepResearch',
    (_event, body: {
      snapshot_id: string
      question: string
      provider_id: string
      model_id: string
      report_id?: string
      max_hops?: number
      include_debug?: boolean
    }) => client.post('/api/qa/deep-research', body)
  )
}
