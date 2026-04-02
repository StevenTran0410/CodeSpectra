import React, { useEffect, useState } from 'react'
import { FolderOpen, Info } from 'lucide-react'

export default function SettingsScreen(): React.ReactElement {
  const [version, setVersion] = useState('')
  const [userDataPath, setUserDataPath] = useState('')

  useEffect(() => {
    window.api.app.getVersion().then(setVersion).catch(() => {})
    window.api.app.getUserDataPath().then(setUserDataPath).catch(() => {})
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
      </div>
    </>
  )
}
