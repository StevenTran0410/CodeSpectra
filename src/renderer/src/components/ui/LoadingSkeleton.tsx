import React from 'react'

interface Props {
  className?: string
}

export function Skeleton({ className = '' }: Props): React.ReactElement {
  return (
    <div className={`animate-pulse rounded bg-surface-overlay ${className}`} />
  )
}

export function CardSkeleton(): React.ReactElement {
  return (
    <div className="card p-4 space-y-3">
      <Skeleton className="h-4 w-1/3" />
      <Skeleton className="h-3 w-2/3" />
      <Skeleton className="h-3 w-1/2" />
    </div>
  )
}
