import Database from 'better-sqlite3'
import { app } from 'electron'
import path from 'path'
import fs from 'fs'
import { logger } from '../logger'

// Migrations run in order. Never edit an existing migration — add a new one.
const MIGRATIONS: Array<{ version: number; sql: string }> = [
  {
    version: 0,
    sql: `
      CREATE TABLE IF NOT EXISTS app_metadata (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS workspaces (
        id          TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL,
        settings    TEXT NOT NULL DEFAULT '{}'
      );

      INSERT OR IGNORE INTO app_metadata VALUES ('schema_version', '0');
      INSERT OR IGNORE INTO app_metadata VALUES ('first_launched_at', datetime('now'));
    `
  }
  // RPA-011 will add tables: repository_connections, repository_snapshots,
  // analysis_runs, report_artifacts, secret_references, consent_records, job_events
]

let db: Database.Database | null = null

export function getDb(): Database.Database {
  if (!db) throw new Error('Database not initialized — call initDb() first')
  return db
}

export function initDb(): Database.Database {
  const dbDir = app.getPath('userData')
  fs.mkdirSync(dbDir, { recursive: true })

  const dbPath = path.join(dbDir, 'codespectra.db')
  logger.info(`Opening database at: ${dbPath}`)

  db = new Database(dbPath)
  db.pragma('journal_mode = WAL')
  db.pragma('foreign_keys = ON')

  runMigrations(db)

  return db
}

function runMigrations(database: Database.Database): void {
  // On first run app_metadata doesn't exist yet — treat as version -1
  const tableExists = database
    .prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='app_metadata'`)
    .get()

  const currentVersion = tableExists
    ? parseInt(
        (
          database
            .prepare(`SELECT value FROM app_metadata WHERE key = 'schema_version'`)
            .get() as { value: string } | undefined
        )?.value ?? '-1',
        10
      )
    : -1

  const pending = MIGRATIONS.filter((m) => m.version > currentVersion)
  if (pending.length === 0) {
    logger.info(`Database schema up to date (version ${currentVersion})`)
    return
  }

  for (const migration of pending) {
    logger.info(`Running migration ${migration.version}`)
    database.exec(migration.sql)
    database
      .prepare(`INSERT OR REPLACE INTO app_metadata VALUES ('schema_version', ?)`)
      .run(String(migration.version))
    logger.info(`Migration ${migration.version} complete`)
  }
}

export function closeDb(): void {
  db?.close()
  db = null
}
