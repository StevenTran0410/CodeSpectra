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
    'qa:classifyIntent',
    (_event, body: { question: string }) => client.post('/api/qa/classify-intent', body)
  )

  ipcMain.handle('qa:classifier:status', () => client.get('/api/qa/classifier/status'))
  ipcMain.handle('qa:classifier:examples', () => client.get('/api/qa/classifier/examples'))
  ipcMain.handle(
    'qa:classifier:addExample',
    (_event, body: { text: string; is_deep_research: boolean }) =>
      client.post('/api/qa/classifier/examples', body)
  )
  ipcMain.handle(
    'qa:classifier:deleteExample',
    (_event, id: string) => client.del(`/api/qa/classifier/examples/${id}`)
  )
  ipcMain.handle('qa:classifier:retrain', () => client.post('/api/qa/classifier/retrain', {}))

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
