import React from 'react'

interface Props {
  message: string
  onBack: () => void
  onSkip: () => void
}

export function SkipWarningModal({ message, onBack, onSkip }: Props): React.ReactElement {
  return (
    <div className="fixed inset-0 z-[110] flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="card w-full max-w-sm p-5 shadow-2xl">
        <p className="text-sm text-gray-300 leading-relaxed">{message}</p>
        <div className="flex gap-2 justify-end mt-5">
          <button className="btn-ghost" onClick={onSkip}>
            Skip for now
          </button>
          <button className="btn-primary" onClick={onBack}>
            Go back
          </button>
        </div>
      </div>
    </div>
  )
}
