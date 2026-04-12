import React from 'react'
import { Spinner } from './Spinner'

interface PageLoadingProps {
  label?: string
}

export const PageLoading: React.FC<PageLoadingProps> = ({ label }) => (
  <div className="h-full flex items-center justify-center flex-col gap-3">
    <Spinner size="lg" />
    {label && <p className="text-sm text-zinc-400">{label}</p>}
  </div>
)
