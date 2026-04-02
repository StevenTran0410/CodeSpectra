import React from 'react'

export type BadgeVariant = 'local' | 'cloud' | 'synced' | 'indexing' | 'failed' | 'idle' | 'cancelled'

interface Props {
  variant: BadgeVariant
  label?: string
  className?: string
}

const CONFIGS: Record<BadgeVariant, { defaultLabel: string; classes: string; pulse?: boolean }> = {
  local: {
    defaultLabel: 'Strict Local',
    classes: 'bg-green-950 text-green-400 border-green-800'
  },
  cloud: {
    defaultLabel: 'BYOK Cloud',
    classes: 'bg-blue-950 text-blue-400 border-blue-800'
  },
  synced: {
    defaultLabel: 'Synced',
    classes: 'bg-teal-950 text-teal-400 border-teal-800'
  },
  indexing: {
    defaultLabel: 'Indexing',
    classes: 'bg-yellow-950 text-yellow-400 border-yellow-800',
    pulse: true
  },
  failed: {
    defaultLabel: 'Failed',
    classes: 'bg-red-950 text-red-400 border-red-800'
  },
  idle: {
    defaultLabel: 'Idle',
    classes: 'bg-surface-overlay text-gray-500 border-surface-border'
  },
  cancelled: {
    defaultLabel: 'Cancelled',
    classes: 'bg-gray-950 text-gray-500 border-gray-800'
  }
}

export function StatusBadge({ variant, label, className = '' }: Props): React.ReactElement {
  const config = CONFIGS[variant]
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium border ${config.classes} ${className}`}
    >
      {config.pulse ? (
        <span className="relative flex h-1.5 w-1.5">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-yellow-400" />
        </span>
      ) : (
        <span className="h-1.5 w-1.5 rounded-full bg-current" />
      )}
      {label ?? config.defaultLabel}
    </span>
  )
}
