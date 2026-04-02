import React, { useState, useEffect, useRef } from 'react'
import { X } from 'lucide-react'

interface Props {
  mode: 'create' | 'rename'
  initialName?: string
  onConfirm: (name: string) => Promise<void>
  onClose: () => void
}

export function WorkspaceModal({ mode, initialName = '', onConfirm, onClose }: Props): React.ReactElement {
  const [name, setName] = useState(initialName)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    inputRef.current?.focus()
    inputRef.current?.select()
  }, [])

  const handleSubmit = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault()
    const trimmed = name.trim()
    if (!trimmed) return

    setLoading(true)
    setError(null)
    try {
      await onConfirm(trimmed)
      onClose()
    } catch (err) {
      setError(String(err))
      setLoading(false)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="card w-full max-w-sm p-5 shadow-2xl">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-semibold text-gray-100">
            {mode === 'create' ? 'New Workspace' : 'Rename Workspace'}
          </h2>
          <button className="btn-ghost p-1" onClick={onClose} disabled={loading}>
            <X className="w-4 h-4" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1.5">Name</label>
            <input
              ref={inputRef}
              className="input"
              placeholder="e.g. My Project"
              value={name}
              onChange={(e) => setName(e.target.value)}
              maxLength={80}
              disabled={loading}
            />
            {error && <p className="text-red-400 text-xs mt-1.5">{error}</p>}
          </div>

          <div className="flex gap-2 justify-end pt-1">
            <button type="button" className="btn-secondary" onClick={onClose} disabled={loading}>
              Cancel
            </button>
            <button
              type="submit"
              className="btn-primary"
              disabled={!name.trim() || loading}
            >
              {loading ? 'Saving…' : mode === 'create' ? 'Create' : 'Rename'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
