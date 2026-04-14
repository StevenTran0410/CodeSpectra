import { ipcMain } from 'electron'
import type { BackendClient } from '../infrastructure/python-server/client'

/** Register a fire-and-forget streaming IPC channel that forwards SSE events to the renderer. */
function registerStreamChannel(channel: string, endpoint: string, client: BackendClient): void {
  ipcMain.on(channel, async (event, body) => {
    try {
      await client.postStream(endpoint, body, (streamEvent) => {
        event.sender.send('qa:stream-event', streamEvent)
      })
    } catch (err) {
      event.sender.send('qa:stream-event', { type: 'error', message: String(err) })
    }
  })
}

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

  // Streaming versions — ipcMain.on (fire-and-forget) so push events never race an invoke reply.
  registerStreamChannel('qa:ask:stream', '/api/qa/ask/stream', client)
  registerStreamChannel('qa:deepResearch:stream', '/api/qa/deep-research/stream', client)

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
