import { spawn, type ChildProcess } from 'child_process'
import path from 'path'
import fs from 'fs'
import http from 'http'
import { app } from 'electron'
import { is } from '@electron-toolkit/utils'
import { logger } from '../../shared/logger'

const STARTUP_TIMEOUT_MS = 15_000
const HEALTH_POLL_INTERVAL_MS = 250
// Keep a rolling buffer of recent stdout/stderr so a startup-timeout error can
// show what (if anything) the process actually printed, instead of a bare
// "timed out" message that gives no diagnostic signal.
const MAX_CAPTURED_OUTPUT_CHARS = 4000

class AEHProcessManager {
  private process: ChildProcess | null = null
  private port: number | null = null
  private isShuttingDown = false
  private capturedOutput = ''
  private spawnError: Error | null = null
  // Tracks an in-flight start() attempt so concurrent callers join the SAME
  // attempt instead of each independently killing the other's spawned process.
  // This is not a theoretical concern — React 18 StrictMode's dev-mode
  // double-effect-invocation calls startAEH() twice in quick succession on
  // mount, which (before this fix) caused the second call to kill the first
  // call's still-starting process and spawn a competing one; the first call's
  // waitForReady() then polled a dead port until it timed out ~15s later,
  // surfacing as a delayed "Failed to launch" error even though the second
  // call's process was already up and being used successfully.
  private startPromise: Promise<number> | null = null

  private appendCapturedOutput(chunk: string): void {
    this.capturedOutput = (this.capturedOutput + chunk).slice(-MAX_CAPTURED_OUTPUT_CHARS)
  }

  async start(): Promise<number> {
    if (this.port !== null) {
      logger.info(`[AEHProcessManager] AEH server already running on port ${this.port}`)
      return this.port
    }

    if (this.startPromise) {
      logger.info('[AEHProcessManager] start() already in flight — joining existing attempt')
      return this.startPromise
    }

    this.startPromise = this.startInternal()
    try {
      return await this.startPromise
    } finally {
      this.startPromise = null
    }
  }

  private async startInternal(): Promise<number> {
    // Clean up any stale/failed-but-still-running process from a prior attempt
    if (this.process && !this.process.killed) {
      logger.info('[AEHProcessManager] Killing stale process from previous attempt...')
      this.process.kill('SIGTERM')
    }
    this.process = null
    this.capturedOutput = ''
    this.spawnError = null

    this.isShuttingDown = false
    const port = await findFreePort()

    const pythonBin = is.dev
      ? path.join(
          process.cwd(),
          'agent_eval_harness',
          '.venv',
          process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python'
        )
      : path.join(process.resourcesPath, 'agent_eval_harness', 'python')

    const dataDir = app.getPath('userData')

    logger.info(`[AEHProcessManager] is.dev=${is.dev} process.cwd()=${process.cwd()}`)
    logger.info(`[AEHProcessManager] Resolved pythonBin=${pythonBin}`)
    logger.info(`[AEHProcessManager] pythonBin exists on disk: ${fs.existsSync(pythonBin)}`)
    logger.info(`[AEHProcessManager] AEH_DATA_DIR=${dataDir}`)
    logger.info(`[AEHProcessManager] Starting AEH server: ${pythonBin} -m agent_eval_harness.cli ui --port ${port}`)

    this.process = spawn(
      pythonBin,
      ['-m', 'agent_eval_harness.cli', 'ui', '--port', String(port)],
      {
        cwd: is.dev ? path.join(process.cwd(), 'agent_eval_harness') : undefined,
        env: {
          ...process.env,
          AEH_DATA_DIR: dataDir,
          AEH_CASES_PER_AGENT: '5', // Stage-3 cases generated per agent — edit this to change (5 = cheap MVP default)
          PYTHONIOENCODING: 'utf-8'
        },
        stdio: ['ignore', 'pipe', 'pipe']
      }
    )

    // Catch spawn-level failures (e.g. ENOENT if pythonBin is wrong) — without
    // this handler, an 'error' event with no listener can go unnoticed while
    // the caller just sees a generic startup timeout with no real cause.
    this.process.on('error', (err) => {
      logger.info(`[AEHProcessManager] Spawn error: ${err.message}`)
      this.spawnError = err
    })

    // Forward output logs to electron logger, and keep a rolling copy for
    // diagnostics if startup times out.
    this.process.stderr?.on('data', (data: Buffer) => {
      const text = data.toString()
      logger.info(`[AEH stderr] ${text.trim()}`)
      this.appendCapturedOutput(`[stderr] ${text}`)
    })

    this.process.stdout?.on('data', (data: Buffer) => {
      const text = data.toString()
      logger.info(`[AEH stdout] ${text.trim()}`)
      this.appendCapturedOutput(`[stdout] ${text}`)
    })

    try {
      // Wait for /health route response
      await this.waitForReady(port)
    } catch (err) {
      // Startup failed — kill what we spawned, don't leave it running, then re-throw
      // with whatever output/spawn-error we captured, so the failure is actually
      // diagnosable instead of a bare "timed out" message.
      if (this.process && !this.process.killed) {
        logger.info('[AEHProcessManager] Killing process after startup failure...')
        this.process.kill('SIGTERM')
      }
      this.process = null
      // Cast explicitly: TS narrows `this.spawnError` to its last known synchronous
      // assignment (the `= null` above) and doesn't account for the `.on('error', ...)`
      // callback reassigning it asynchronously while `await this.waitForReady(port)`
      // was suspended — without this cast it incorrectly infers `never` here.
      const currentSpawnError = this.spawnError as Error | null
      const spawnErrorMessage: string | null = currentSpawnError ? currentSpawnError.message : null
      const capturedOutput: string = this.capturedOutput
      let detail: string
      if (spawnErrorMessage) {
        detail = `spawn error: ${spawnErrorMessage}`
      } else if (capturedOutput) {
        detail = `captured output before timeout:\n${capturedOutput}`
      } else {
        detail = 'no stdout/stderr was captured before timing out (process produced zero output)'
      }
      logger.info(`[AEHProcessManager] Startup failure detail — ${detail}`)
      throw new Error(`${(err as Error).message} — ${detail}`)
    }

    this.port = port
    logger.info(`[AEHProcessManager] AEH server ready on port ${port}`)

    this.process.on('exit', (code, signal) => {
      logger.info(`[AEHProcessManager] AEH server exited: code=${code} signal=${signal}`)
      this.process = null
      this.port = null
    })

    return port
  }

  private async waitForReady(port: number): Promise<void> {
    const startTime = Date.now()

    return new Promise<void>((resolve, reject) => {
      const poll = async () => {
        if (this.isShuttingDown) {
          reject(new Error('AEH server shutdown requested during startup'))
          return
        }

        const isReady = await this.checkHealth(port)
        if (isReady) {
          resolve()
          return
        }

        if (Date.now() - startTime > STARTUP_TIMEOUT_MS) {
          reject(new Error(`AEH server startup timed out after ${STARTUP_TIMEOUT_MS}ms`))
          return
        }

        setTimeout(poll, HEALTH_POLL_INTERVAL_MS)
      }

      poll()
    })
  }

  private async checkHealth(port: number): Promise<boolean> {
    return new Promise((resolve) => {
      const req = http.get(`http://127.0.0.1:${port}/health`, { timeout: 1000 }, (res) => {
        resolve(res.statusCode === 200)
      })
      req.on('timeout', () => {
        req.destroy()
        resolve(false)
      })
      req.on('error', () => {
        resolve(false)
      })
      req.end()
    })
  }

  stop(): void {
    this.isShuttingDown = true
    if (this.process && !this.process.killed) {
      logger.info('[AEHProcessManager] Stopping AEH server...')
      this.process.kill('SIGTERM')
    }
    this.process = null
    this.port = null
  }

  getPort(): number | null {
    return this.port
  }
}

// Singleton instance
const aehProcessManager = new AEHProcessManager()

export function getAEHProcessManager(): AEHProcessManager {
  return aehProcessManager
}

async function findFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const net = require('net') as typeof import('net')
    const server = net.createServer()
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address()
      if (!addr || typeof addr === 'string') return reject(new Error('Could not get port'))
      const port = addr.port
      server.close(() => resolve(port))
    })
    server.on('error', reject)
  })
}
