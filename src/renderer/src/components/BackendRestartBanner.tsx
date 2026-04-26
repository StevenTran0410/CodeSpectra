import { useEffect, useState } from 'react'
import { useAppBackendStore, attachBackendListeners } from '../store/app-backend.store'

export function BackendRestartBanner(): JSX.Element | null {
  const [visible, setVisible] = useState(false)
  const backendStatus = useAppBackendStore((state) => state.backendStatus)
  const restartAttempt = useAppBackendStore((state) => state.restartAttempt)
  const maxAttempts = 4

  useEffect(() => {
    attachBackendListeners()
    setVisible(backendStatus === 'restarting')
  }, [backendStatus])

  const handleRetry = async (): Promise<void> => {
    try {
      await window.api.app.retryBackend()
    } catch (error) {
      console.error('[BackendRestartBanner] Retry failed:', error)
    }
  }

  if (!visible) {
    return null
  }

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-amber-50 border-b border-amber-200 px-4 py-3 shadow-sm">
      <div className="flex items-center justify-between max-w-full">
        <div className="flex items-center gap-3">
          <div className="flex-shrink-0">
            <svg
              className="h-5 w-5 text-amber-600 animate-spin"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
          </div>
          <div>
            <p className="text-sm font-medium text-amber-900">
              Backend restarting
              <span className="ml-1 text-amber-700">
                (attempt {restartAttempt}/{maxAttempts})
              </span>
            </p>
          </div>
        </div>
        <button
          onClick={handleRetry}
          className="flex-shrink-0 inline-flex items-center px-3 py-1.5 rounded-md text-sm font-medium bg-amber-100 text-amber-900 hover:bg-amber-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-amber-500 transition-colors"
        >
          Retry now
        </button>
      </div>
    </div>
  )
}
