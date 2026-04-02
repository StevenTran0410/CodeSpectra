import { AlertTriangle } from 'lucide-react'

interface ErrorBannerProps {
  message: string
  onDismiss?: () => void
  className?: string
}

/**
 * Dismissible inline error banner.
 * Used whenever a store/API operation fails and we need to surface the message to the user.
 */
export function ErrorBanner({ message, onDismiss, className = '' }: ErrorBannerProps) {
  return (
    <div
      className={`flex items-center gap-2 bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-3 text-sm text-red-400 ${className}`}
    >
      <AlertTriangle size={14} className="shrink-0" />
      <span className="flex-1">{message}</span>
      {onDismiss && (
        <button
          onClick={onDismiss}
          className="text-red-500/60 hover:text-red-400 transition-colors ml-1"
          aria-label="Dismiss error"
        >
          ✕
        </button>
      )}
    </div>
  )
}
