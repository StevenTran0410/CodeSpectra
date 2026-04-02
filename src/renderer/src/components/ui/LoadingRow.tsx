import { Loader2 } from 'lucide-react'

interface LoadingRowProps {
  message?: string
  size?: number
  className?: string
}

/**
 * Inline loading indicator row — spinner + optional text.
 * Used inside sections while async data is being fetched.
 */
export function LoadingRow({
  message = 'Loading…',
  size = 14,
  className = '',
}: LoadingRowProps) {
  return (
    <div className={`flex items-center gap-2 text-sm text-zinc-500 py-4 ${className}`}>
      <Loader2 size={size} className="animate-spin" />
      {message}
    </div>
  )
}
