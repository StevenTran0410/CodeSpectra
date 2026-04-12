import React from 'react'

type BadgeVariant = 'success' | 'warning' | 'error' | 'info' | 'neutral'

interface BadgeProps {
  variant?: BadgeVariant
  size?: 'sm' | 'md'
  dot?: boolean
  className?: string
  children: React.ReactNode
}

const variantMap: Record<BadgeVariant, string> = {
  success: 'bg-emerald-900/40 text-emerald-300 border-emerald-800',
  warning: 'bg-amber-900/40 text-amber-200 border-amber-800',
  error: 'bg-red-900/40 text-red-300 border-red-800',
  info: 'bg-sky-900/50 text-sky-300 border-sky-800',
  neutral: 'bg-zinc-800 text-zinc-300 border-zinc-700'
}

const sizeMap: Record<'sm' | 'md', string> = {
  sm: 'text-[10px] px-1.5 py-0.5',
  md: 'text-xs px-2 py-0.5'
}

export const Badge: React.FC<BadgeProps> = ({
  variant = 'neutral',
  size = 'md',
  dot = false,
  className = '',
  children
}) => {
  const variantClass = variantMap[variant]
  const sizeClass = sizeMap[size]
  const finalClass = `inline-flex items-center gap-1 rounded border font-medium ${variantClass} ${sizeClass} ${className}`.trim()

  return (
    <span className={finalClass}>
      {dot && <span className="h-1.5 w-1.5 rounded-full bg-current" />}
      {children}
    </span>
  )
}
