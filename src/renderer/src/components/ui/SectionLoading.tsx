import React from 'react'
import { Spinner } from './Spinner'

interface SectionLoadingProps {
  label?: string
}

export const SectionLoading: React.FC<SectionLoadingProps> = ({ label }) => (
  <div className="flex items-center gap-2 text-xs text-zinc-500">
    <Spinner size="sm" />
    {label && <span>{label}</span>}
  </div>
)
