/** HTTP client for communicating with the Python FastAPI backend. */
export class BackendClient {
  private readonly base: string

  constructor(port: number) {
    this.base = `http://127.0.0.1:${port}`
  }

  /** Extract a human-readable message from a failed response.
   *  FastAPI returns `{ "detail": "..." }` for 4xx errors — prefer that over raw text. */
  private async _errorMessage(res: Response): Promise<string> {
    try {
      const json = await res.json()
      if (typeof json?.detail === 'string') return json.detail
      if (typeof json?.message === 'string') return json.message
      return JSON.stringify(json)
    } catch {
      return res.text().catch(() => res.statusText)
    }
  }

  async get<T>(path: string): Promise<T> {
    const res = await fetch(`${this.base}${path}`)
    if (!res.ok) throw new Error(await this._errorMessage(res))
    return res.json() as Promise<T>
  }

  async post<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(`${this.base}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
    if (!res.ok) throw new Error(await this._errorMessage(res))
    return res.json() as Promise<T>
  }

  async put<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(`${this.base}${path}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
    if (!res.ok) throw new Error(await this._errorMessage(res))
    return res.json() as Promise<T>
  }

  async del(path: string): Promise<void> {
    const res = await fetch(`${this.base}${path}`, { method: 'DELETE' })
    if (!res.ok) throw new Error(await this._errorMessage(res))
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
