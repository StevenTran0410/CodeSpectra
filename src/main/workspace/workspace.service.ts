import type Database from 'better-sqlite3'
import type { Workspace } from './workspace.types'
import { logger } from '../logger'

interface WorkspaceRow {
  id: string
  name: string
  created_at: string
  updated_at: string
  settings: string
}

function rowToWorkspace(row: WorkspaceRow): Workspace {
  return {
    id: row.id,
    name: row.name,
    created_at: row.created_at,
    updated_at: row.updated_at,
    settings: JSON.parse(row.settings) as Record<string, unknown>
  }
}

export class WorkspaceService {
  constructor(private db: Database.Database) {}

  list(): Workspace[] {
    const rows = this.db
      .prepare('SELECT * FROM workspaces ORDER BY created_at ASC')
      .all() as WorkspaceRow[]
    return rows.map(rowToWorkspace)
  }

  getById(id: string): Workspace | null {
    const row = this.db
      .prepare('SELECT * FROM workspaces WHERE id = ?')
      .get(id) as WorkspaceRow | undefined
    return row ? rowToWorkspace(row) : null
  }

  create(name: string): Workspace {
    const trimmed = name.trim()
    if (!trimmed) throw new Error('Workspace name cannot be empty')

    const id = crypto.randomUUID()
    const now = new Date().toISOString()

    this.db
      .prepare(
        `INSERT INTO workspaces (id, name, created_at, updated_at, settings)
         VALUES (?, ?, ?, ?, '{}')`
      )
      .run(id, trimmed, now, now)

    logger.info(`Workspace created: ${id} "${trimmed}"`)

    return this.getById(id)!
  }

  rename(id: string, name: string): Workspace {
    const trimmed = name.trim()
    if (!trimmed) throw new Error('Workspace name cannot be empty')

    const now = new Date().toISOString()
    const result = this.db
      .prepare('UPDATE workspaces SET name = ?, updated_at = ? WHERE id = ?')
      .run(trimmed, now, id)

    if (result.changes === 0) throw new Error(`Workspace not found: ${id}`)
    logger.info(`Workspace renamed: ${id} → "${trimmed}"`)

    return this.getById(id)!
  }

  delete(id: string): void {
    const result = this.db.prepare('DELETE FROM workspaces WHERE id = ?').run(id)
    if (result.changes === 0) throw new Error(`Workspace not found: ${id}`)
    logger.info(`Workspace deleted: ${id}`)
  }
}
