import path from 'path'
import { promises as fs } from 'fs'
import { app, dialog, ipcMain } from 'electron'
import type { BackendClient } from '../infrastructure/python-server/client'

export function registerFolderHandlers(client: BackendClient): void {
  /** Open native folder-picker dialog and return the selected path (or null if cancelled). */
  ipcMain.handle('folder:pick', async (): Promise<string | null> => {
    const result = await dialog.showOpenDialog({
      title: 'Select Repository Folder',
      defaultPath: app.getPath('home'),
      properties: ['openDirectory'],
      buttonLabel: 'Open Folder',
    })
    if (result.canceled || result.filePaths.length === 0) return null
    return result.filePaths[0]
  })

  ipcMain.handle(
    'folder:validate',
    (_event, path: string) => client.post('/api/local-repo/validate', { path })
  )

  ipcMain.handle(
    'folder:list',
    () => client.get('/api/local-repo/')
  )

  ipcMain.handle(
    'folder:add',
    (_event, path: string) => client.post('/api/local-repo/', { path })
  )

  ipcMain.handle(
    'folder:remove',
    (_event, id: string) => client.del(`/api/local-repo/${id}`)
  )

  ipcMain.handle(
    'folder:revalidate',
    (_event, id: string) => client.post(`/api/local-repo/${id}/revalidate`, {})
  )

  ipcMain.handle(
    'folder:branches',
    (_event, id: string, refresh = false): Promise<string[]> =>
      client.get(`/api/local-repo/${id}/branches?refresh=${refresh ? 'true' : 'false'}`)
  )

  ipcMain.handle(
    'folder:setBranch',
    (_event, id: string, branch: string) =>
      client.post(`/api/local-repo/${id}/branch`, { branch })
  )

  ipcMain.handle(
    'folder:setActiveSnapshot',
    (_event, id: string, snapshotId: string | null) =>
      client.post(`/api/local-repo/${id}/active-snapshot`, { snapshot_id: snapshotId })
  )

  ipcMain.handle(
    'folder:updateSettings',
    (_event, id: string, settings: {
      sync_mode: 'latest' | 'pinned'
      pinned_ref: string | null
      ignore_overrides: string[]
      detect_submodules: boolean
    }) =>
      client.post(`/api/local-repo/${id}/settings`, settings)
  )

  ipcMain.handle(
    'folder:estimateFileCount',
    (_event, id: string) => client.get(`/api/local-repo/${id}/estimate-file-count`)
  )

  /**
   * Clone a remote git URL.
   * Destination is auto-resolved to ~/CodeSpectra/repos/<repo-name>.
   * Returns the registered LocalRepo on success.
   */
  ipcMain.handle('folder:cloneFromUrl', async (_event, url: string) => {
    const repoName = url.split('/').pop()?.replace(/\.git$/, '') ?? 'repo'
    const destPath = path.join(app.getPath('home'), 'CodeSpectra', 'repos', repoName)
    return client.post('/api/local-repo/clone', { url, dest_path: destPath })
  })

  ipcMain.handle(
    'sync:prepare',
    (_event, body: {
      local_repo_id: string
      branch?: string | null
      clone_policy?: 'full' | 'shallow' | 'partial'
    }) => client.post('/api/sync/prepare', body)
  )

  ipcMain.handle(
    'sync:listForRepo',
    (_event, repoId: string) => client.get(`/api/sync/repo/${repoId}`)
  )

  ipcMain.handle(
    'sync:getSnapshot',
    (_event, snapshotId: string) => client.get(`/api/sync/snapshot/${snapshotId}`)
  )

  ipcMain.handle(
    'sync:deleteSnapshot',
    (_event, snapshotId: string) => client.del(`/api/sync/snapshot/${snapshotId}`)
  )

  ipcMain.handle('manifest:build', (_event, snapshotId: string, manualIgnores?: string[]) =>
    client.post('/api/manifest/build', {
      snapshot_id: snapshotId,
      ...(manualIgnores !== undefined ? { manual_ignores: manualIgnores } : {}),
    })
  )

  ipcMain.handle('manifest:tree', (_event, snapshotId: string) =>
    client.get(`/api/manifest/tree/${snapshotId}`)
  )

  ipcMain.handle('manifest:file', (_event, snapshotId: string, relPath: string) =>
    client.get(`/api/manifest/file/${snapshotId}?path=${encodeURIComponent(relPath)}`)
  )

  ipcMain.handle('repomap:build', (_event, snapshotId: string, forceRebuild = true) =>
    client.post('/api/repo-map/build', { snapshot_id: snapshotId, force_rebuild: forceRebuild })
  )

  ipcMain.handle('repomap:summary', (_event, snapshotId: string) =>
    client.get(`/api/repo-map/summary/${snapshotId}`)
  )

  ipcMain.handle('repomap:symbols', (_event, snapshotId: string, limit = 300, pathPrefix?: string) =>
    client.get(
      `/api/repo-map/symbols/${snapshotId}?limit=${limit}${
        pathPrefix ? `&path_prefix=${encodeURIComponent(pathPrefix)}` : ''
      }`
    )
  )

  ipcMain.handle('repomap:search', (_event, snapshotId: string, q: string, limit = 120) =>
    client.get(`/api/repo-map/search/${snapshotId}?q=${encodeURIComponent(q)}&limit=${limit}`)
  )

  ipcMain.handle('repomap:exportCsv', async (_event, snapshotId: string, excludeTests = true) => {
    const data = await client.get<{ snapshot_id: string; row_count: number; csv: string }>(
      `/api/repo-map/export/${snapshotId}?exclude_tests=${excludeTests ? 'true' : 'false'}`
    )
    const defaultName = `codespectra-index-${snapshotId.slice(0, 8)}.csv`
    const save = await dialog.showSaveDialog({
      title: 'Save Deep Index CSV',
      defaultPath: path.join(app.getPath('desktop'), defaultName),
      filters: [{ name: 'CSV', extensions: ['csv'] }],
    })
    if (save.canceled || !save.filePath) {
      return { saved: false, file_path: null, row_count: data.row_count }
    }
    await fs.writeFile(save.filePath, data.csv, 'utf-8')
    return { saved: true, file_path: save.filePath, row_count: data.row_count }
  })

  ipcMain.handle('graph:build', (_event, snapshotId: string, forceRebuild = true) =>
    client.post('/api/graph/build', { snapshot_id: snapshotId, force_rebuild: forceRebuild })
  )

  ipcMain.handle('graph:summary', async (_event, snapshotId: string) => {
    try {
      return await client.get(`/api/graph/summary/${snapshotId}`)
    } catch (err) {
      const msg = err instanceof Error ? err.message.toLowerCase() : String(err).toLowerCase()
      if (msg.includes('not found')) {
        return null
      }
      throw err
    }
  })

  ipcMain.handle('graph:edges', (_event, snapshotId: string, limit = 2000) =>
    client.get(`/api/graph/edges/${snapshotId}?limit=${limit}`)
  )

  ipcMain.handle('graph:neighbors', (_event, snapshotId: string, seedPath: string, hops = 1, limit = 300) =>
    client.get(
      `/api/graph/neighbors/${snapshotId}?path=${encodeURIComponent(seedPath)}&hops=${hops}&limit=${limit}`
    )
  )

  ipcMain.handle('retrieval:buildIndex', (_event, snapshotId: string, forceRebuild = true) =>
    client.post('/api/retrieval/build-index', { snapshot_id: snapshotId, force_rebuild: forceRebuild })
  )

  ipcMain.handle(
    'retrieval:retrieve',
    (_event, body: {
      snapshot_id: string
      query: string
      section: 'architecture' | 'conventions' | 'feature_map' | 'important_files' | 'glossary'
      mode?: 'hybrid' | 'vectorless'
      max_results?: number
    }) => client.post('/api/retrieval/retrieve', body)
  )

  ipcMain.handle(
    'retrieval:compare',
    (_event, body: {
      snapshot_id: string
      query: string
      section: 'architecture' | 'conventions' | 'feature_map' | 'important_files' | 'glossary'
      max_results?: number
    }) => client.post('/api/retrieval/compare', body)
  )

  ipcMain.handle(
    'analysis:estimate',
    (_event, repoId: string, snapshotId: string) =>
      client.get(`/api/analysis/estimate/${repoId}/${snapshotId}`)
  )

  ipcMain.handle(
    'analysis:start',
    async (
      event,
      body: {
        repo_id: string
        snapshot_id: string
        scan_mode: 'quick' | 'full'
        privacy_mode: 'strict_local' | 'byok_cloud'
        provider_id: string
        model_id: string
      }
    ) => {
      const job = await client.post<{ id: string }>('/api/analysis/start', body)
      const sender = event.sender
      let fromIdx = 0
      const timer = setInterval(async () => {
        if (sender.isDestroyed()) {
          clearInterval(timer)
          return
        }
        try {
          const res = await client.get<{ events: unknown[]; job_done: boolean }>(
            `/api/analysis/events/${job.id}?from_idx=${fromIdx}`
          )
          fromIdx += res.events.length
          for (const evt of res.events) {
            sender.send('analysis:section_done', evt)
          }
          if (res.job_done) clearInterval(timer)
        } catch {
          // transient error — continue polling
        }
      }, 2000)
      return job
    }
  )

  ipcMain.handle(
    'analysis:listReports',
    (_event, repoId?: string, limit = 30) =>
      client.get(`/api/analysis/reports?${repoId ? `repo_id=${encodeURIComponent(repoId)}&` : ''}limit=${limit}`)
  )

  ipcMain.handle(
    'analysis:getReport',
    (_event, reportId: string) => client.get(`/api/analysis/reports/${reportId}`)
  )

  ipcMain.handle(
    'analysis:getReportByJob',
    (_event, jobId: string) => client.get(`/api/analysis/report-by-job/${jobId}`)
  )

  ipcMain.handle(
    'analysis:deleteReport',
    (_event, reportId: string) => client.del(`/api/analysis/reports/${reportId}`)
  )

  ipcMain.handle('analysis:exportReportMarkdown', async (_event, reportId: string) => {
    const data = await client.get<{ report_id: string; default_name: string; markdown: string }>(
      `/api/analysis/reports/${reportId}/markdown`
    )
    const save = await dialog.showSaveDialog({
      title: 'Save Analysis Report Markdown',
      defaultPath: path.join(app.getPath('desktop'), data.default_name),
      filters: [{ name: 'Markdown', extensions: ['md'] }],
    })
    if (save.canceled || !save.filePath) {
      return { saved: false, file_path: null }
    }
    await fs.writeFile(save.filePath, data.markdown, 'utf-8')
    return { saved: true, file_path: save.filePath }
  })

  // backward-compat typo alias used by some renderer builds
  ipcMain.handle(
    'analysis:deleteRepot',
    (_event, reportId: string) => client.del(`/api/analysis/reports/${reportId}`)
  )

  // ── Git / SSH settings ────────────────────────────────────────────────────
  ipcMain.handle('git:getConfig', () => client.get('/api/app/git-config'))

  ipcMain.handle('git:setConfig', (_event, sshKeyPath: string | null) =>
    client.put('/api/app/git-config', { ssh_key_path: sshKeyPath })
  )

  ipcMain.handle('git:pickSshKey', async (): Promise<string | null> => {
    const result = await dialog.showOpenDialog({
      title: 'Select SSH Private Key',
      defaultPath: path.join(app.getPath('home'), '.ssh'),
      properties: ['openFile'],
      buttonLabel: 'Use This Key',
      filters: [
        { name: 'SSH Keys', extensions: ['*'] },
      ],
    })
    if (result.canceled || result.filePaths.length === 0) return null
    return result.filePaths[0]
  })
}
