import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useWorkspaceStore } from '../../store/workspace.store'
import { SkipWarningModal } from './SkipWarningModal'

interface Props {
  onDone: () => void
}

const SKIP_MESSAGES: Record<1 | 2, string> = {
  1: "Without a workspace you won't be able to connect repositories or run analysis. You can create one later from the Home screen.",
  2: "Without a provider the app can't run analysis. You can set one up later from the Providers screen."
}

export function OnboardingWizard({ onDone }: Props): React.ReactElement {
  const [step, setStep] = useState<1 | 2>(1)
  const [showSkipWarning, setShowSkipWarning] = useState(false)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const create = useWorkspaceStore((s) => s.create)
  const navigate = useNavigate()

  const handleCreateWorkspace = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault()
    const trimmedName = name.trim()
    if (!trimmedName) return

    setLoading(true)
    setError(null)
    try {
      await create(trimmedName, description.trim() || undefined)
      setStep(2)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  const handleSetUpProvider = (): void => {
    onDone()
    navigate('/providers')
  }

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-surface">
      <div className="w-full max-w-md px-6">
        <div className="mb-8 text-center">
          <span className="text-2xl font-bold text-gray-100">CodeSpectra</span>
          <p className="text-sm text-gray-500 mt-1">Step {step} of 2</p>
        </div>

        {step === 1 && (
          <div className="card p-6 shadow-2xl">
            <h2 className="text-base font-semibold text-gray-100 mb-1">Create your first workspace</h2>
            <p className="text-sm text-gray-400 mb-5">
              Workspaces group your repository connections and analysis runs.
            </p>

            <form onSubmit={handleCreateWorkspace} className="space-y-4">
              <div>
                <label className="block text-xs text-gray-400 mb-1.5">Name</label>
                <input
                  className="input"
                  placeholder="e.g. My Project"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  maxLength={80}
                  disabled={loading}
                  autoFocus
                />
                {error && <p className="text-red-400 text-xs mt-1.5">{error}</p>}
              </div>

              <div>
                <label className="block text-xs text-gray-400 mb-1.5">
                  Description <span className="text-gray-600">(optional)</span>
                </label>
                <textarea
                  className="input resize-none"
                  placeholder="What is this workspace for?"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  maxLength={200}
                  rows={2}
                  disabled={loading}
                />
              </div>

              <div className="flex items-center justify-between pt-1">
                <button
                  type="button"
                  className="text-sm text-gray-500 hover:text-gray-300 transition-colors"
                  onClick={() => setShowSkipWarning(true)}
                  disabled={loading}
                >
                  Set up later
                </button>
                <button
                  type="submit"
                  className="btn-primary"
                  disabled={!name.trim() || loading}
                >
                  {loading ? 'Creating…' : 'Create workspace'}
                </button>
              </div>
            </form>
          </div>
        )}

        {step === 2 && (
          <div className="card p-6 shadow-2xl">
            <h2 className="text-base font-semibold text-gray-100 mb-1">Set up a provider</h2>
            <p className="text-sm text-gray-400 mb-5">
              Providers are AI models (local or cloud) that power CodeSpectra's analysis engine.
              Connect at least one to start scanning repositories.
            </p>

            <div className="flex items-center justify-between pt-1">
              <button
                type="button"
                className="text-sm text-gray-500 hover:text-gray-300 transition-colors"
                onClick={() => setShowSkipWarning(true)}
              >
                Set up later
              </button>
              <button className="btn-primary" onClick={handleSetUpProvider}>
                Set up Provider
              </button>
            </div>
          </div>
        )}
      </div>

      {showSkipWarning && (
        <SkipWarningModal
          message={SKIP_MESSAGES[step]}
          onBack={() => setShowSkipWarning(false)}
          onSkip={() => {
            setShowSkipWarning(false)
            if (step === 1) {
              onDone()
            } else {
              onDone()
            }
          }}
        />
      )}
    </div>
  )
}
