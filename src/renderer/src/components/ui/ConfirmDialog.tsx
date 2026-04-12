import React, { useState } from 'react'
import { Modal } from './Modal'
import { Button } from './Button'

interface ConfirmDialogProps {
  open: boolean
  onClose: () => void
  onConfirm: () => void | Promise<void>
  title: string
  description: React.ReactNode
  confirmLabel?: string
  cancelLabel?: string
  confirmVariant?: 'danger' | 'primary'
  loading?: boolean
}

export const ConfirmDialog: React.FC<ConfirmDialogProps> = ({
  open,
  onClose,
  onConfirm,
  title,
  description,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  confirmVariant = 'danger',
  loading = false
}) => {
  const [isLoading, setIsLoading] = useState(false)

  const handleConfirm = async () => {
    setIsLoading(true)
    try {
      await onConfirm()
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Modal open={open} onClose={onClose}>
      <Modal.Overlay onClick={onClose}>
        <Modal.Panel className="max-w-sm">
          <Modal.Header title={title} onClose={onClose} />
          <div className="px-5 py-4">
            <div className="text-sm text-gray-400">{description}</div>
          </div>
          <Modal.Footer>
            <Button
              variant="secondary"
              size="md"
              onClick={onClose}
              disabled={isLoading || loading}
            >
              {cancelLabel}
            </Button>
            <Button
              variant={confirmVariant}
              size="md"
              loading={isLoading || loading}
              onClick={handleConfirm}
            >
              {confirmLabel}
            </Button>
          </Modal.Footer>
        </Modal.Panel>
      </Modal.Overlay>
    </Modal>
  )
}
