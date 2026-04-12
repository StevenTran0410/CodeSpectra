import React from 'react'
import { Loader2 } from 'lucide-react'

interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

const sizeMap: Record<'sm' | 'md' | 'lg', number> = {
  sm: 12,
  md: 16,
  lg: 24
}

export const Spinner: React.FC<SpinnerProps> = ({ size = 'md', className = '' }) => {
  const px = sizeMap[size]
  return (
    <Loader2
      size={px}
      className={`animate-spin ${className}`}
    />
  )
}
