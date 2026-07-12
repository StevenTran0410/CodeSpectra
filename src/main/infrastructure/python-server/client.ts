/** Thrown by BackendClient on a non-2xx response; `status` lets callers branch on 404 without parsing message text. */
export class BackendHttpError extends Error {
  constructor(message: string, public readonly status: number) {
    super(message)
  }
}

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
      const json: unknown = await res.json()
      if (json && typeof json === 'object' && 'detail' in json && typeof (json as { detail: unknown }).detail === 'string') {
        return (json as { detail: string }).detail
      }
      if (json && typeof json === 'object' && 'message' in json && typeof (json as { message: unknown }).message === 'string') {
        return (json as { message: string }).message
      }
      return JSON.stringify(json)
    } catch {
      return res.text().catch(() => res.statusText)
    }
  }

  private async _throwHttpError(res: Response): Promise<never> {
    throw new BackendHttpError(await this._errorMessage(res), res.status)
  }

  async get<T>(path: string): Promise<T> {
    const res = await fetch(`${this.base}${path}`)
    if (!res.ok) await this._throwHttpError(res)
    return res.json() as Promise<T>
  }

  /** timeoutMs overrides undici's default 5-minute headers timeout — pass a larger value for routes that run many sequential LLM calls server-side (e.g. dataset fulfillment). */
  async post<T>(path: string, body: unknown, timeoutMs?: number): Promise<T> {
    const res = await fetch(`${this.base}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: timeoutMs ? AbortSignal.timeout(timeoutMs) : undefined
    })
    if (!res.ok) await this._throwHttpError(res)
    return res.json() as Promise<T>
  }

  async put<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(`${this.base}${path}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
    if (!res.ok) await this._throwHttpError(res)
    return res.json() as Promise<T>
  }

  /** Most DELETE routes return 204 (nothing to parse); a 200 + JSON body route gets it parsed and typed. */
  async del<T = void>(path: string): Promise<T> {
    const res = await fetch(`${this.base}${path}`, { method: 'DELETE' })
    if (!res.ok) await this._throwHttpError(res)
    if (res.status === 204) return undefined as T
    const text = await res.text()
    return (text ? JSON.parse(text) : undefined) as T
  }

  /** Consume a Server-Sent Events stream. Calls `onEvent` for every parsed data frame.
   *  Resolves when the stream ends or an error/done event is received. */
  async postStream(
    path: string,
    body: unknown,
    onEvent: (event: Record<string, unknown>) => void
  ): Promise<void> {
    const res = await fetch(`${this.base}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) throw new Error(await this._errorMessage(res))

    const reader = res.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          const data = JSON.parse(line.slice(6)) as Record<string, unknown>
          onEvent(data)
        } catch {
          // ignore malformed frames
        }
      }
    }
  }

  async healthCheck(): Promise<boolean> {
    try {
      const res = await fetch(`${this.base}/api/app/health`, { signal: AbortSignal.timeout(2000) })
      return res.ok
    } catch {
      return false
    }
  }
}
