import React, { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAEHStore } from '../../store/aeh.store'
import { AlertCircle } from 'lucide-react'

export default function AEHScreen(): React.ReactElement {
  const containerRef = useRef<HTMLDivElement>(null)
  const navigate = useNavigate()
  const { port, isLoading, error, startAEH, showView, hideView, resizeView, setActive } =
    useAEHStore()
  const [started, setStarted] = useState(false)

  // Track active state in store for sidebar highlights
  useEffect(() => {
    setActive(true)
    return () => setActive(false)
  }, [setActive])

  // Trigger backend start
  useEffect(() => {
    startAEH()
      .then(() => setStarted(true))
      .catch(() => {})
  }, [startAEH])

  // View lifecycle and resizing overlay bounds
  useEffect(() => {
    if (isLoading || error || !started || port === null || !containerRef.current) return

    const getPixelBounds = () => {
      if (!containerRef.current) return null
      const rect = containerRef.current.getBoundingClientRect()
      return {
        x: Math.round(rect.x),
        y: Math.round(rect.y),
        width: Math.round(rect.width),
        height: Math.round(rect.height)
      }
    }

    const initView = async () => {
      const bounds = getPixelBounds()
      if (bounds) {
        await showView(bounds)
      }
    }

    initView()

    // Listen to resize events
    const updateBounds = () => {
      const bounds = getPixelBounds()
      if (bounds) {
        resizeView(bounds).catch(() => {})
      }
    }

    window.addEventListener('resize', updateBounds)

    const observer = new ResizeObserver(() => {
      updateBounds()
    })
    observer.observe(containerRef.current)

    return () => {
      window.removeEventListener('resize', updateBounds)
      observer.disconnect()
      hideView().catch(() => {})
    }
  }, [port, isLoading, error, started, showView, hideView, resizeView])

  return (
    <div className="w-full h-full min-h-[calc(100vh-32px)] bg-[#090d16] flex flex-col p-4">
      {/* Renderer Header with Back navigation */}
      <div className="h-12 flex items-center justify-between border-b border-[#1f2937] mb-2 bg-[#090d16] z-10 shrink-0">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-bold text-indigo-400 tracking-wider uppercase">
            AEH Interactive Dashboard
          </span>
        </div>
        <button
          onClick={() => navigate('/')}
          className="px-4 py-1.5 bg-slate-900 border border-slate-800 text-xs rounded hover:bg-slate-800 hover:border-slate-700 transition-colors text-slate-300 font-medium"
        >
          Back to CodeSpectra
        </button>
      </div>

      {/* Frame placeholder target where WebContentsView attaches */}
      <div
        ref={containerRef}
        className="flex-1 w-full bg-slate-950 rounded-xl border border-slate-800/80 relative overflow-hidden"
      >
        {isLoading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center space-y-4">
            <div className="h-10 w-10 border-t-2 border-indigo-500 rounded-full animate-spin"></div>
            <p className="text-xs text-slate-400 font-medium">Starting AEH Dashboard backend...</p>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center space-y-3 p-6 text-center">
            <AlertCircle className="h-8 w-8 text-red-500" />
            <p className="text-sm font-semibold text-red-400">Failed to launch evaluation dashboard</p>
            <p className="text-xs text-slate-500 max-w-sm">{error}</p>
          </div>
        )}

        {!isLoading && !error && !started && (
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <p className="text-xs text-slate-500 italic">Initializing UI layer...</p>
          </div>
        )}
      </div>
    </div>
  )
}
