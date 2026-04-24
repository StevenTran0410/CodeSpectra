import React, { useEffect, useRef } from 'react'
import { X } from 'lucide-react'

interface ModalProps {
  open: boolean
  onClose: () => void
  children: React.ReactNode
  className?: string
}

interface ModalOverlayProps {
  onClick?: (e: React.MouseEvent<HTMLDivElement>) => void
  children: React.ReactNode
}

interface ModalPanelProps {
  className?: string
  children: React.ReactNode
}

interface ModalHeaderProps {
  title: string
  onClose?: () => void
}

interface ModalFooterProps {
  children: React.ReactNode
}

const ModalOverlay: React.FC<ModalOverlayProps> = ({ onClick, children }) => (
  <div
    role="dialog"
    aria-modal="true"
    aria-labelledby="modal-title"
    className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center"
    onClick={onClick}
  >
    {children}
  </div>
)

const ModalPanel: React.FC<ModalPanelProps> = ({ className = 'max-w-lg', children }) => (
  <div
    className={`relative w-full mx-4 shadow-2xl rounded-lg bg-surface animate-modal-in ${className}`}
    onClick={(e) => e.stopPropagation()}
  >
    {children}
  </div>
)

const ModalHeader: React.FC<ModalHeaderProps> = ({ title, onClose }) => (
  <div className="flex items-center justify-between p-5 border-b border-surface-border">
    <h2 id="modal-title" className="text-base font-semibold text-gray-100">{title}</h2>
    {onClose && (
      <button
        type="button"
        onClick={onClose}
        className="text-gray-400 hover:text-gray-200 transition-colors"
        aria-label="Close modal"
      >
        <X size={20} />
      </button>
    )}
  </div>
)

const ModalFooter: React.FC<ModalFooterProps> = ({ children }) => (
  <div className="flex items-start justify-end gap-2 px-5 py-4 border-t border-surface-border">
    {children}
  </div>
)

const Modal: React.FC<ModalProps> & {
  Overlay: typeof ModalOverlay
  Panel: typeof ModalPanel
  Header: typeof ModalHeader
  Footer: typeof ModalFooter
} = ({ open, onClose, children, className }) => {
  const panelRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [open, onClose])

  useEffect(() => {
    if (!open || !panelRef.current) return

    const focusableElements = panelRef.current.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    )
    const firstElement = focusableElements[0] as HTMLElement
    if (firstElement) {
      firstElement.focus()
    }

    const handleTabKey = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return

      const focusableList = Array.from(focusableElements) as HTMLElement[]
      // Guard for empty-children edge case: no focusable elements in modal
      if (focusableList.length === 0) return

      const currentIndex = focusableList.indexOf(document.activeElement as HTMLElement)
      let nextIndex = currentIndex + 1

      if (e.shiftKey) {
        nextIndex = currentIndex - 1
        if (nextIndex < 0) nextIndex = focusableList.length - 1
      } else {
        if (nextIndex >= focusableList.length) nextIndex = 0
      }

      e.preventDefault()
      focusableList[nextIndex]?.focus()
    }

    document.addEventListener('keydown', handleTabKey)
    return () => document.removeEventListener('keydown', handleTabKey)
  }, [open])

  if (!open) return null

  return (
    <ModalOverlay onClick={onClose}>
      <div ref={panelRef} className={className}>
        {children}
      </div>
    </ModalOverlay>
  )
}

Modal.Overlay = ModalOverlay
Modal.Panel = ModalPanel
Modal.Header = ModalHeader
Modal.Footer = ModalFooter

export { Modal }
