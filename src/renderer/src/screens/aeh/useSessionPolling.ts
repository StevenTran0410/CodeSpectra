import { useCallback, useEffect, useRef } from 'react'

interface PollableSession {
  id: string
  status: string
}

interface UseSessionPollingOptions<T extends PollableSession> {
  fetchSession: (sessionId: string) => Promise<T>
  onUpdate: (session: T) => void
  /** Called once when status leaves 'running'; do stage-specific follow-up work here. */
  onDone: (session: T) => void | Promise<void>
  onError?: () => void
  intervalMs?: number
}

/** Polls a session endpoint on an interval until its status leaves 'running'; cleans up on unmount. */
export function useSessionPolling<T extends PollableSession>(
  options: UseSessionPollingOptions<T>
): (sessionId: string) => void {
  const optionsRef = useRef(options)
  optionsRef.current = options

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const startPolling = useCallback(
    (sessionId: string) => {
      stopPolling()
      pollRef.current = setInterval(async () => {
        try {
          const session = await optionsRef.current.fetchSession(sessionId)
          optionsRef.current.onUpdate(session)
          if (session.status !== 'running') {
            stopPolling()
            await optionsRef.current.onDone(session)
          }
        } catch (err) {
          stopPolling()
          optionsRef.current.onError?.()
        }
      }, optionsRef.current.intervalMs ?? 2000)
    },
    [stopPolling]
  )

  useEffect(() => stopPolling, [stopPolling])

  return startPolling
}
