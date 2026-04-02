/** HTTP client for communicating with the Python FastAPI backend. */
export class BackendClient {
  private readonly base: string

  constructor(port: number) {
    this.base = `http://127.0.0.1:${port}`
  }

  async get<T>(path: string): Promise<T> {
    const res = await fetch(`${this.base}${path}`)
    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText)
      throw new Error(`Backend GET ${path} failed (${res.status}): ${text}`)
    }
    return res.json() as Promise<T>
  }

  async post<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(`${this.base}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText)
      throw new Error(`Backend POST ${path} failed (${res.status}): ${text}`)
    }
    return res.json() as Promise<T>
  }

  async put<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(`${this.base}${path}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText)
      throw new Error(`Backend PUT ${path} failed (${res.status}): ${text}`)
    }
    return res.json() as Promise<T>
  }

  async del(path: string): Promise<void> {
    const res = await fetch(`${this.base}${path}`, { method: 'DELETE' })
    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText)
      throw new Error(`Backend DELETE ${path} failed (${res.status}): ${text}`)
    }
  }

  async healthCheck(): Promise<boolean> {
    try {
      const res = await fetch(`${this.base}/health`, { signal: AbortSignal.timeout(2000) })
      return res.ok
    } catch {
      return false
    }
  }
}
