import { AlertTriangle, Cloud, ExternalLink, X } from 'lucide-react'

interface ConsentBannerProps {
  onAccept: () => void
  onDismiss: () => void
}

export function ConsentBanner({ onAccept, onDismiss }: ConsentBannerProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="relative w-full max-w-lg bg-zinc-900 border border-amber-500/30 rounded-2xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center gap-3 px-6 pt-6 pb-4 border-b border-zinc-800">
          <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center">
            <Cloud size={20} className="text-amber-400" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-zinc-100">Cloud Provider Notice</h2>
            <p className="text-xs text-zinc-500 mt-0.5">Read before connecting a cloud AI service</p>
          </div>
          <button
            onClick={onDismiss}
            className="ml-auto p-1.5 text-zinc-600 hover:text-zinc-300 hover:bg-zinc-800 rounded-lg transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-5 space-y-4">
          <div className="flex items-start gap-3 bg-amber-500/5 border border-amber-500/20 rounded-xl px-4 py-3">
            <AlertTriangle size={16} className="text-amber-400 shrink-0 mt-0.5" />
            <p className="text-sm text-amber-200/80 leading-relaxed">
              When using a cloud provider, your <strong className="text-amber-200">code and repository context</strong> will
              be sent to external AI services over the internet.
            </p>
          </div>

          <div className="space-y-2">
            <p className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">What this means</p>
            <ul className="space-y-2 text-sm text-zinc-400">
              {[
                'Source code snippets will leave your device',
                'Data is processed under each provider\'s privacy policy',
                'API keys are stored in this app\'s local database',
                'No data is sent until you start an analysis run',
              ].map((item) => (
                <li key={item} className="flex items-start gap-2">
                  <span className="shrink-0 mt-1.5 w-1.5 h-1.5 rounded-full bg-zinc-600" />
                  {item}
                </li>
              ))}
            </ul>
          </div>

          <div className="bg-zinc-800/60 rounded-xl px-4 py-3 text-xs text-zinc-500">
            <strong className="text-zinc-400">Do not use cloud providers</strong> with confidential, proprietary,
            or NDA-covered code. Use <span className="text-emerald-400">Strict Local mode</span> (Ollama / LM Studio)
            for sensitive repositories.
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center gap-3 px-6 pb-6">
          <button
            onClick={onDismiss}
            className="flex-1 py-2.5 text-sm text-zinc-400 hover:text-zinc-200 border border-zinc-700 hover:border-zinc-600 rounded-xl transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onAccept}
            className="flex-1 py-2.5 text-sm font-medium bg-amber-500 hover:bg-amber-400 text-zinc-900 rounded-xl transition-colors"
          >
            I understand — proceed
          </button>
        </div>
      </div>
    </div>
  )
}
