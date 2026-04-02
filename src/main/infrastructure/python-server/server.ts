import { spawn, type ChildProcess } from 'child_process'
import path from 'path'
import { app } from 'electron'
import { is } from '@electron-toolkit/utils'
import { logger } from '../../shared/logger'
import { BackendClient } from './client'

const STARTUP_TIMEOUT_MS = 20_000
const HEALTH_POLL_INTERVAL_MS = 300
const READY_SIGNAL = 'BACKEND_READY:'

let pythonProcess: ChildProcess | null = null
let backendPort: number | null = null

export async function startPythonServer(): Promise<BackendClient> {
  const port = await findFreePort()
  const scriptPath = is.dev
    ? path.join(process.cwd(), 'backend', 'main.py')
    : path.join(process.resourcesPath, 'backend', 'main.py')

  // In dev use the venv Python so all dependencies are available.
  // In prod use the bundled binary (PyInstaller output).
  const pythonBin = is.dev
    ? path.join(
        process.cwd(),
        'backend',
        '.venv',
        process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python'
      )
    : path.join(process.resourcesPath, 'backend', 'python')

  const dataDir = app.getPath('userData')

  logger.info(`Starting Python backend: ${pythonBin} ${scriptPath} --port ${port}`)

  pythonProcess = spawn(pythonBin, [scriptPath, '--port', String(port)], {
    env: {
      ...process.env,
      CODESPECTRA_DATA_DIR: dataDir,
      CODESPECTRA_ENV: is.dev ? 'development' : 'production'
    },
    stdio: ['ignore', 'pipe', 'pipe']
  })

  // Forward Python logs to Electron logger
  pythonProcess.stderr?.on('data', (data: Buffer) => {
    logger.info(`[Python] ${data.toString().trim()}`)
  })

  // Wait for BACKEND_READY signal on stdout
  const readyPromise = new Promise<void>((resolve, reject) => {
    let buffer = ''
    pythonProcess!.stdout?.on('data', (data: Buffer) => {
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
    pythonProcess!.on('error', reject)
    pythonProcess!.on('exit', (code) => {
      reject(new Error(`Python process exited with code ${code} before becoming ready`))
    })
  })

  const timeoutPromise = new Promise<never>((_, reject) =>
    setTimeout(() => reject(new Error(`Python backend startup timed out after ${STARTUP_TIMEOUT_MS}ms`)), STARTUP_TIMEOUT_MS)
  )

  await Promise.race([readyPromise, timeoutPromise])

  backendPort = port
  logger.info(`Python backend ready on port ${port}`)

  // Handle unexpected crashes after startup
  pythonProcess.on('exit', (code, signal) => {
    if (code !== 0 && signal !== 'SIGTERM') {
      logger.error(`Python backend crashed: exit=${code} signal=${signal}`)
    }
  })

  return new BackendClient(port)
}

export function stopPythonServer(): void {
  if (pythonProcess && !pythonProcess.killed) {
    logger.info('Stopping Python backend...')
    pythonProcess.kill('SIGTERM')
    pythonProcess = null
    backendPort = null
  }
}

export function getBackendPort(): number | null {
  return backendPort
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
