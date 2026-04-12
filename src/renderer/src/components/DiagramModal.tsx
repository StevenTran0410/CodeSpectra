import React, { useCallback } from 'react'
import { X } from 'lucide-react'
import type { C4DiagramData } from '../types/analysis'
import C4DiagramView from './C4DiagramView'

interface DiagramModalProps {
  open: boolean
  onClose: () => void
  diagramData: C4DiagramData
  title: string
}

export default function DiagramModal({
  open,
  onClose,
  diagramData,
  title,
}: DiagramModalProps): React.ReactElement | null {
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    },
    [onClose]
  )

  if (!open) return null

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={title}
      className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex flex-col"
      onKeyDown={handleKeyDown}
      tabIndex={-1}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800 bg-zinc-950 shrink-0">
        <span className="text-sm font-medium text-zinc-200">{title}</span>
        <button
          onClick={onClose}
          className="flex items-center justify-center w-7 h-7 rounded hover:bg-zinc-800 text-zinc-400 hover:text-zinc-100 transition-colors"
          aria-label="Close diagram"
        >
          <X size={16} />
        </button>
      </div>

      {/* Diagram fills remaining space — React Flow has built-in zoom/pan/controls */}
      <div className="flex-1" style={{ minHeight: 0 }}>
        <C4DiagramView
          data={diagramData}
          className="w-full h-full"
          style={{ height: '100%', borderRadius: 0 }}
        />
      </div>
    </div>
  )
}
