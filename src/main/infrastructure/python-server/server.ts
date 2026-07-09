import { spawn, type ChildProcess } from 'child_process'
import crypto from 'crypto'
import path from 'path'
import { app, BrowserWindow } from 'electron'
import { is } from '@electron-toolkit/utils'
import { logger } from '../../shared/logger'
import { BackendClient } from './client'

const STARTUP_TIMEOUT_MS = 60_000  // Bumped from 20s for slower machines
const HEALTH_POLL_INTERVAL_MS = 300
const READY_SIGNAL = 'BACKEND_READY:'

// Shared secret so AEH's subprocess can authenticate to require_external_token.
const externalApiToken = crypto.randomBytes(32).toString('hex')

export function getExternalApiToken(): string {
  return externalApiToken
}

// PythonProcessManager: encapsulates process lifecycle with crash recovery

class PythonProcessManager {
  private process: ChildProcess | null = null
  private port: number | null = null
  private client: BackendClient | null = null
  private isShuttingDown = false
  private restartAttempt = 0
  private maxRestartAttempts = 4

  async start(): Promise<BackendClient> {
    this.isShuttingDown = false
    this.restartAttempt = 0
    const port = await findFreePort()
    const scriptPath = is.dev
      ? path.join(process.cwd(), 'backend', 'main.py')
      : path.join(process.resourcesPath, 'backend', 'main.py')

    const pythonBin = is.dev
      ? path.join(
          process.cwd(),
          'backend',
          '.venv',
          process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python'
        )
      : path.join(process.resourcesPath, 'backend', 'python')

    const dataDir = app.getPath('userData')

    logger.info(`[PythonProcessManager] Starting Python backend: ${pythonBin} ${scriptPath} --port ${port}`)

    this.process = spawn(pythonBin, [scriptPath, '--port', String(port)], {
      env: {
        ...process.env,
        CODESPECTRA_DATA_DIR: dataDir,
        CODESPECTRA_ENV: is.dev ? 'development' : 'production',
        CODESPECTRA_EXTERNAL_TOKEN: externalApiToken
      },
      stdio: ['ignore', 'pipe', 'pipe']
    })

    // Forward Python logs
    this.process.stderr?.on('data', (data: Buffer) => {
      logger.info(`[Python] ${data.toString().trim()}`)
    })

    // Wait for BACKEND_READY signal
    const readyPromise = new Promise<void>((resolve, reject) => {
      let buffer = ''
      this.process!.stdout?.on('data', (data: Buffer) => {
        buffer += data.toString()
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''
        for (const line of lines) {
          logger.info(`[Python stdout] ${line}`)
          if (line.startsWith(READY_SIGNAL)) {
            resolve()
          }
        }
      })
      this.process!.on('error', reject)
      this.process!.on('exit', (code) => {
        reject(new Error(`Python process exited with code ${code} before becoming ready`))
      })
    })

    const timeoutPromise = new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error(`Python backend startup timed out after ${STARTUP_TIMEOUT_MS}ms`)), STARTUP_TIMEOUT_MS)
    )

    await Promise.race([readyPromise, timeoutPromise])

    this.port = port
    this.client = new BackendClient(port)
    logger.info(`[PythonProcessManager] Python backend ready on port ${port}`)

    // Emit ready event to renderer
    const mainWindow = BrowserWindow.getAllWindows()[0]
    if (mainWindow) {
      mainWindow.webContents.send('backend:ready')
    }

    // Handle crashes after startup
    this.process.on('exit', (code, signal) => {
      this.onCrash(code, signal)
    })

    return this.client
  }

  private onCrash(code: number | null, signal: string | null): void {
    if (this.isShuttingDown) {
      return
    }

    logger.error(`[PythonProcessManager] Python backend crashed: exit=${code} signal=${signal}`)
    this.client = null

    // Emit crash event to renderer
    const mainWindow = BrowserWindow.getAllWindows()[0]
    if (mainWindow) {
      mainWindow.webContents.send('app:backend-died', { attempt: this.restartAttempt + 1 })
    }

    // Schedule restart with exponential backoff
    this.restartWithBackoff()
  }

  private restartWithBackoff(): void {
    if (this.restartAttempt >= this.maxRestartAttempts) {
      logger.error(`[PythonProcessManager] Max restart attempts (${this.maxRestartAttempts}) reached`)
      return
    }

    this.restartAttempt++
    const delayMs = 1000 * Math.pow(2, this.restartAttempt - 1)

    logger.info(`[PythonProcessManager] Scheduling restart attempt ${this.restartAttempt}/${this.maxRestartAttempts} in ${delayMs}ms`)

    setTimeout(() => {
      if (this.isShuttingDown) {
        return
      }
      logger.info(`[PythonProcessManager] Attempting restart ${this.restartAttempt}/${this.maxRestartAttempts}`)
      this.start().catch((err) => {
        logger.error(`[PythonProcessManager] Restart failed: ${err.message}`)
      })
    }, delayMs)
  }

  stop(): void {
    this.isShuttingDown = true
    if (this.process && !this.process.killed) {
      logger.info('[PythonProcessManager] Stopping Python backend...')
      this.process.kill('SIGTERM')
    }
    this.process = null
    this.port = null
    this.client = null
  }

  getClient(): BackendClient | null {
    return this.client
  }

  getPort(): number | null {
    return this.port
  }
}

// Singleton instance
const processManager = new PythonProcessManager()

// Backwards compatibility exports
let pythonProcess: ChildProcess | null = null
let backendPort: number | null = null

export async function startPythonServer(): Promise<BackendClient> {
  return await processManager.start()
}

export function stopPythonServer(): void {
  processManager.stop()
}

export function getBackendPort(): number | null {
  return processManager.getPort()
}

export function getPythonProcessManager(): PythonProcessManager {
  return processManager
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
