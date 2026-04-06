import React, { useEffect, useState } from 'react'
import { FolderOpen, Info, Cpu } from 'lucide-react'

interface NativeFunction {
  name: string
  available: boolean
  description: string
}

interface Diagnostics {
  python_version: string
  native_module_loaded: boolean
  native_functions: NativeFunction[]
}

export default function SettingsScreen(): React.ReactElement {
  const [version, setVersion] = useState('')
  const [userDataPath, setUserDataPath] = useState('')
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null)
  const [diagError, setDiagError] = useState<string | null>(null)

  useEffect(() => {
    window.api.app.getVersion().then(setVersion).catch(() => {})
    window.api.app.getUserDataPath().then(setUserDataPath).catch(() => {})
    window.api.app.getDiagnostics()
      .then(setDiagnostics)
      .catch((e) => setDiagError(String(e)))
  }, [])

  return (
    <>
      <div className="screen-header">
        <h1 className="screen-title">Settings</h1>
        <p className="screen-subtitle">App configuration, privacy defaults, and storage</p>
      </div>

      <div className="p-6 space-y-6 max-w-2xl">
        {/* App info */}
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

        {/* Privacy defaults placeholder */}
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

        {/* Storage placeholder */}
        <section className="card p-4 space-y-3">
          <h2 className="text-sm font-medium text-gray-300 flex items-center gap-2">
            <FolderOpen className="w-4 h-4" />
            Storage
          </h2>
          <p className="text-xs text-gray-500">
            Cache path, max cache size, and auto-cleanup settings will be configurable here in RPA-011.
          </p>
        </section>

        {/* Native engine diagnostics */}
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

          {diagnostics && (
            <div className="space-y-2 text-sm">
              {/* overall status */}
              <div className="flex justify-between items-center py-1.5 border-b border-surface-border">
                <span className="text-gray-400">Module status</span>
                {diagnostics.native_module_loaded ? (
                  <span className="text-xs text-green-400 bg-green-950 border border-green-800 px-2 py-0.5 rounded-full">
                    ✓ Loaded
                  </span>
                ) : (
                  <span className="text-xs text-red-400 bg-red-950 border border-red-800 px-2 py-0.5 rounded-full">
                    ✗ Not built — using Python fallback
                  </span>
                )}
              </div>
              <div className="flex justify-between py-1.5 border-b border-surface-border">
                <span className="text-gray-400">Python version</span>
                <span className="text-gray-200 font-mono text-xs">{diagnostics.python_version}</span>
              </div>

              {/* per-function table */}
              <div className="pt-1 space-y-1">
                {diagnostics.native_functions.map((fn) => (
                  <div key={fn.name} className="flex items-start gap-2 py-1 border-b border-surface-border last:border-0">
                    <span className={`mt-0.5 shrink-0 text-[11px] font-mono px-1.5 py-0.5 rounded border ${
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

              {!diagnostics.native_module_loaded && (
                <p className="text-[11px] text-yellow-500 pt-1">
                  Run <span className="font-mono bg-zinc-800 px-1 rounded">python scripts/build_native_graph.py</span> inside <span className="font-mono">backend/</span> with the x64 VS Developer Command Prompt to enable C++ acceleration.
                </p>
              )}
            </div>
          )}
        </section>
      </div>
    </>
  )
}
