import React, { useEffect, useState } from 'react'
import { FolderOpen, Info, Cpu, BrainCircuit, Plus, Trash2, RefreshCw, CheckCircle, XCircle } from 'lucide-react'

interface NativeFunction {
  name: string
  available: boolean
  description: string
  module: string
}

interface Diagnostics {
  python_version: string
  native_module_loaded: boolean
  native_functions: NativeFunction[]
}

function NativeModulePopup({
  functions,
  onClose,
}: {
  functions: NativeFunction[]
  onClose: () => void
}): React.ReactElement {
  const available = functions.filter((f) => f.available).length
  const total = functions.length

  // Group by module
  const byModule: Record<string, NativeFunction[]> = {}
  for (const fn of functions) {
    const mod = fn.module || 'unknown'
    if (!byModule[mod]) byModule[mod] = []
    byModule[mod].push(fn)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div
        className="bg-surface-overlay border border-surface-border rounded-xl shadow-2xl w-[480px] max-h-[70vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="px-5 py-4 border-b border-surface-border flex items-center justify-between shrink-0">
          <div>
            <h3 className="text-sm font-semibold text-gray-100">Native C++ Modules</h3>
            <p className="text-xs text-gray-500 mt-0.5">
              {available}/{total} functions accelerated
            </p>
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-300 transition-colors text-lg leading-none">&times;</button>
        </div>

        {/* Scrollable body */}
        <div className="overflow-y-auto px-5 py-3 space-y-4">
          {Object.entries(byModule).map(([mod, fns]) => {
            const modAvail = fns.filter((f) => f.available).length
            return (
              <div key={mod}>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-mono text-gray-400">{mod}</span>
                  <span className={`text-[10px] px-1.5 py-0.5 rounded-full border ${
                    modAvail === fns.length
                      ? 'text-green-400 bg-green-950 border-green-800'
                      : modAvail > 0
                      ? 'text-yellow-400 bg-yellow-950 border-yellow-800'
                      : 'text-red-400 bg-red-950 border-red-800'
                  }`}>
                    {modAvail}/{fns.length}
                  </span>
                </div>
                <div className="space-y-1">
                  {fns.map((fn) => (
                    <div key={fn.name} className="flex items-start gap-2 py-1.5 border-b border-surface-border/50 last:border-0">
                      <span className={`mt-0.5 shrink-0 text-[10px] font-mono px-1.5 py-0.5 rounded border ${
                        fn.available
                          ? 'text-green-400 bg-green-950 border-green-800'
                          : 'text-yellow-400 bg-yellow-950 border-yellow-800'
                      }`}>
                        {fn.available ? 'C++' : 'PY'}
                      </span>
                      <div className="min-w-0">
                        <div className="text-xs font-mono text-gray-200">{fn.name}</div>
                        <div className="text-[11px] text-gray-500 leading-4">{fn.description}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )
          })}
        </div>

        {/* Footer hint */}
        {functions.some((f) => !f.available) && (
          <div className="px-5 py-3 border-t border-surface-border shrink-0">
            <p className="text-[11px] text-yellow-500">
              Run <span className="font-mono bg-zinc-800 px-1 rounded">python scripts/build_native_graph.py</span> (builds graph + BM25) and <span className="font-mono bg-zinc-800 px-1 rounded">python scripts/build_native_chunker.py</span> inside <span className="font-mono">backend/</span> to enable C++ acceleration.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

interface ClassifierStatus {
  trained: boolean
  backend: string
  builtin_examples: number
  user_examples: number
}

interface ClassifierExample {
  id: string
  text: string
  is_deep_research: boolean
  created_at: string
}

export default function SettingsScreen(): React.ReactElement {
  const [version, setVersion] = useState('')
  const [userDataPath, setUserDataPath] = useState('')
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null)
  const [diagError, setDiagError] = useState<string | null>(null)
  const [showNativePopup, setShowNativePopup] = useState(false)

  // Classifier state
  const [classifierStatus, setClassifierStatus] = useState<ClassifierStatus | null>(null)
  const [classifierExamples, setClassifierExamples] = useState<ClassifierExample[]>([])
  const [classifierLoading, setClassifierLoading] = useState(false)
  const [classifierError, setClassifierError] = useState<string | null>(null)
  const [retraining, setRetraining] = useState(false)
  const [showAddForm, setShowAddForm] = useState(false)
  const [newExampleText, setNewExampleText] = useState('')
  const [newExampleLabel, setNewExampleLabel] = useState<boolean>(true)
  const [addingExample, setAddingExample] = useState(false)

  useEffect(() => {
    window.api.app.getVersion().then(setVersion).catch(() => {})
    window.api.app.getUserDataPath().then(setUserDataPath).catch(() => {})
    window.api.app.getDiagnostics()
      .then(setDiagnostics)
      .catch((e) => setDiagError(String(e)))
    loadClassifier()
  }, [])

  const loadClassifier = async () => {
    setClassifierLoading(true)
    setClassifierError(null)
    try {
      const [status, examples] = await Promise.all([
        window.api.qa.classifier.status(),
        window.api.qa.classifier.examples(),
      ])
      setClassifierStatus(status)
      setClassifierExamples(examples)
    } catch (e) {
      setClassifierError(String(e))
    } finally {
      setClassifierLoading(false)
    }
  }

  const handleRetrain = async () => {
    setRetraining(true)
    setClassifierError(null)
    try {
      const status = await window.api.qa.classifier.retrain()
      setClassifierStatus(status)
    } catch (e) {
      setClassifierError(String(e))
    } finally {
      setRetraining(false)
    }
  }

  const handleAddExample = async () => {
    const text = newExampleText.trim()
    if (!text) return
    setAddingExample(true)
    setClassifierError(null)
    try {
      const example = await window.api.qa.classifier.addExample({ text, is_deep_research: newExampleLabel })
      setClassifierExamples((prev) => [example, ...prev])
      setClassifierStatus((s) => s ? { ...s, user_examples: s.user_examples + 1 } : s)
      setNewExampleText('')
      setShowAddForm(false)
    } catch (e) {
      setClassifierError(String(e))
    } finally {
      setAddingExample(false)
    }
  }

  const handleDeleteExample = async (id: string) => {
    setClassifierError(null)
    try {
      await window.api.qa.classifier.deleteExample(id)
      setClassifierExamples((prev) => prev.filter((e) => e.id !== id))
      setClassifierStatus((s) => s ? { ...s, user_examples: Math.max(0, s.user_examples - 1) } : s)
    } catch (e) {
      setClassifierError(String(e))
    }
  }

  const backendLabel = (b: string) => {
    if (b === 'sklearn') return { text: 'sklearn (TF-IDF + LR)', color: 'text-green-400 bg-green-950 border-green-800' }
    if (b === 'pure_python') return { text: 'Pure Python (Naïve Bayes)', color: 'text-yellow-400 bg-yellow-950 border-yellow-800' }
    return { text: 'Not trained', color: 'text-gray-400 bg-surface-raised border-surface-border' }
  }

  return (
    <>
      <div className="screen-header">
        <h1 className="screen-title">Settings</h1>
        <p className="screen-subtitle">App configuration, privacy defaults, and storage</p>
      </div>

      <div className="p-6 space-y-6 max-w-2xl">
        <section className="card p-4 space-y-3">
          <h2 className="text-sm font-medium text-gray-300 flex items-center gap-2">
            <Info className="w-4 h-4" />
            Application
          </h2>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between py-1.5 border-b border-surface-border">
              <span className="text-gray-400">Version</span>
              <span className="text-gray-200 font-mono">{version || '—'}</span>
            </div>
            <div className="flex justify-between items-start py-1.5 border-b border-surface-border gap-4">
              <span className="text-gray-400 shrink-0">User data path</span>
              <span className="text-gray-200 font-mono text-xs text-right break-all">
                {userDataPath || '—'}
              </span>
            </div>
          </div>
        </section>

        <section className="card p-4 space-y-3">
          <h2 className="text-sm font-medium text-gray-300">Privacy defaults</h2>
          <p className="text-xs text-gray-500">
            Default privacy mode and provider settings will be configurable here in RPA-011.
          </p>
          <div className="flex items-center justify-between py-2 border border-surface-border rounded-md px-3">
            <span className="text-sm text-gray-400">Default mode</span>
            <span className="text-xs text-green-400 bg-green-950 border border-green-800 px-2 py-0.5 rounded-full">
              Strict Local
            </span>
          </div>
        </section>

        <section className="card p-4 space-y-3">
          <h2 className="text-sm font-medium text-gray-300 flex items-center gap-2">
            <FolderOpen className="w-4 h-4" />
            Storage
          </h2>
          <p className="text-xs text-gray-500">
            Cache path, max cache size, and auto-cleanup settings will be configurable here in RPA-011.
          </p>
        </section>

        {/* ── Deep Research Classifier ─────────────────────────────────────── */}
        <section className="card p-4 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-gray-300 flex items-center gap-2">
              <BrainCircuit className="w-4 h-4 text-purple-400" />
              Deep Research Classifier
            </h2>
            <div className="flex items-center gap-2">
              <button
                onClick={handleRetrain}
                disabled={retraining || classifierLoading}
                className="flex items-center gap-1.5 px-2.5 py-1 text-xs text-zinc-400 hover:text-zinc-200 hover:bg-surface-overlay rounded transition-colors disabled:opacity-50"
              >
                <RefreshCw size={12} className={retraining ? 'animate-spin' : ''} />
                {retraining ? 'Retraining…' : 'Retrain'}
              </button>
              <button
                onClick={() => { setShowAddForm(true); setNewExampleText(''); setNewExampleLabel(true) }}
                className="flex items-center gap-1.5 px-2.5 py-1 text-xs bg-purple-700 hover:bg-purple-600 text-white rounded transition-colors"
              >
                <Plus size={12} />
                Add Example
              </button>
            </div>
          </div>

          <p className="text-xs text-gray-500">
            Local ML model that detects whether a question needs deep research. Runs entirely on-device
            with no API calls. Add labelled examples to improve accuracy for your domain and language.
          </p>

          {classifierError && (
            <p className="text-xs text-red-400 bg-red-950/30 border border-red-800/40 rounded px-3 py-2">
              {classifierError}
            </p>
          )}

          {/* Status row */}
          {classifierLoading && !classifierStatus && (
            <p className="text-xs text-gray-500 animate-pulse">Loading…</p>
          )}

          {classifierStatus && (() => {
            const bl = backendLabel(classifierStatus.backend)
            return (
              <div className="space-y-2 text-sm">
                <div className="flex justify-between items-center py-1.5 border-b border-surface-border">
                  <span className="text-gray-400">Status</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full border ${classifierStatus.trained ? 'text-green-400 bg-green-950 border-green-800' : 'text-gray-400 bg-surface-raised border-surface-border'}`}>
                    {classifierStatus.trained ? '✓ Trained' : 'Not trained'}
                  </span>
                </div>
                <div className="flex justify-between items-center py-1.5 border-b border-surface-border">
                  <span className="text-gray-400">Backend</span>
                  <span className={`text-xs font-mono px-2 py-0.5 rounded border ${bl.color}`}>{bl.text}</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-surface-border">
                  <span className="text-gray-400">Built-in examples</span>
                  <span className="text-gray-200 font-mono text-xs">{classifierStatus.builtin_examples.toLocaleString()}</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-surface-border">
                  <span className="text-gray-400">Your examples</span>
                  <span className="text-gray-200 font-mono text-xs">{classifierStatus.user_examples}</span>
                </div>
              </div>
            )
          })()}

          {/* Add example form */}
          {showAddForm && (
            <div className="border border-purple-500/30 bg-purple-950/10 rounded-lg p-3 space-y-3">
              <p className="text-xs font-medium text-purple-300">New labelled example</p>
              <textarea
                value={newExampleText}
                onChange={(e) => setNewExampleText(e.target.value)}
                placeholder="Type a question that would be asked about a codebase…"
                rows={2}
                className="w-full px-3 py-2 bg-surface-overlay border border-surface-border rounded text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-purple-500 resize-none"
              />
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-400">Label:</span>
                <button
                  onClick={() => setNewExampleLabel(true)}
                  className={`flex items-center gap-1 px-2.5 py-1 text-xs rounded border transition-colors ${newExampleLabel ? 'bg-purple-700 border-purple-500 text-white' : 'border-surface-border text-gray-400 hover:text-gray-200'}`}
                >
                  <BrainCircuit size={11} />
                  Deep Research
                </button>
                <button
                  onClick={() => setNewExampleLabel(false)}
                  className={`flex items-center gap-1 px-2.5 py-1 text-xs rounded border transition-colors ${!newExampleLabel ? 'bg-blue-700 border-blue-500 text-white' : 'border-surface-border text-gray-400 hover:text-gray-200'}`}
                >
                  Quick Ask
                </button>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={handleAddExample}
                  disabled={!newExampleText.trim() || addingExample}
                  className="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 text-white text-xs rounded transition-colors disabled:opacity-50"
                >
                  {addingExample ? 'Adding…' : 'Add & Retrain'}
                </button>
                <button
                  onClick={() => setShowAddForm(false)}
                  className="px-3 py-1.5 text-gray-400 hover:text-gray-200 text-xs rounded transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* User examples list */}
          {classifierExamples.length > 0 && (
            <div className="space-y-1">
              <p className="text-xs text-gray-500 mb-2">Your examples ({classifierExamples.length})</p>
              {classifierExamples.map((ex) => (
                <div
                  key={ex.id}
                  className="flex items-start gap-2 py-1.5 px-2 rounded hover:bg-surface-overlay group"
                >
                  {ex.is_deep_research ? (
                    <CheckCircle size={13} className="shrink-0 mt-0.5 text-purple-400" />
                  ) : (
                    <XCircle size={13} className="shrink-0 mt-0.5 text-blue-400" />
                  )}
                  <span className="flex-1 text-xs text-gray-300 leading-4">{ex.text}</span>
                  <span className={`shrink-0 text-[10px] px-1.5 py-0.5 rounded border ${ex.is_deep_research ? 'text-purple-400 bg-purple-950 border-purple-800' : 'text-blue-400 bg-blue-950 border-blue-800'}`}>
                    {ex.is_deep_research ? 'deep' : 'quick'}
                  </span>
                  <button
                    onClick={() => handleDeleteExample(ex.id)}
                    className="shrink-0 opacity-0 group-hover:opacity-100 text-gray-500 hover:text-red-400 transition-all"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="card p-4 space-y-3">
          <h2 className="text-sm font-medium text-gray-300 flex items-center gap-2">
            <Cpu className="w-4 h-4" />
            Native Engine (C++)
          </h2>

          {diagError && (
            <p className="text-xs text-red-400">Backend not reachable: {diagError}</p>
          )}

          {!diagnostics && !diagError && (
            <p className="text-xs text-gray-500 animate-pulse">Loading…</p>
          )}

          {diagnostics && (() => {
            const avail = diagnostics.native_functions.filter((f) => f.available).length
            const total = diagnostics.native_functions.length
            const allLoaded = avail === total
            return (
              <div className="space-y-2 text-sm">
                <div className="flex justify-between py-1.5 border-b border-surface-border">
                  <span className="text-gray-400">Python version</span>
                  <span className="text-gray-200 font-mono text-xs">{diagnostics.python_version}</span>
                </div>
                <div className="flex justify-between items-center py-1.5">
                  <span className="text-gray-400">C++ acceleration</span>
                  <button
                    onClick={() => setShowNativePopup(true)}
                    className={`text-xs px-2.5 py-1 rounded-full border cursor-pointer hover:brightness-125 transition-all ${
                      allLoaded
                        ? 'text-green-400 bg-green-950 border-green-800'
                        : avail > 0
                        ? 'text-yellow-400 bg-yellow-950 border-yellow-800'
                        : 'text-red-400 bg-red-950 border-red-800'
                    }`}
                  >
                    {avail}/{total} modules
                  </button>
                </div>
              </div>
            )
          })()}

          {showNativePopup && diagnostics && (
            <NativeModulePopup
              functions={diagnostics.native_functions}
              onClose={() => setShowNativePopup(false)}
            />
          )}
        </section>
      </div>
    </>
  )
}
