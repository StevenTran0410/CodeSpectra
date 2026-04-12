import React, { useEffect, useState } from 'react'
import { CheckCircle2, XCircle, Info, AlertTriangle, X } from 'lucide-react'
import { useToastStore, type Toast } from '../../store/toast.store'

const iconMap: Record<string, React.ReactNode> = {
  success: <CheckCircle2 size={18} />,
  error: <XCircle size={18} />,
  info: <Info size={18} />,
  warning: <AlertTriangle size={18} />
}

const colorMap: Record<string, string> = {
  success: 'bg-emerald-900/30 border-emerald-700 text-emerald-300',
  error: 'bg-red-900/30 border-red-700 text-red-300',
  info: 'bg-sky-900/30 border-sky-700 text-sky-300',
  warning: 'bg-amber-900/30 border-amber-700 text-amber-200'
}

const ToastItem: React.FC<{ toast: Toast }> = ({ toast }) => {
  const { dismiss } = useToastStore()
  const [isExiting, setIsExiting] = useState(false)

  const handleDismiss = () => {
    setIsExiting(true)
    setTimeout(() => {
      dismiss(toast.id)
    }, 200)
  }

  return (
    <div
      className={`rounded-lg border px-4 py-3 text-sm flex items-start gap-2 shadow-lg w-72 transition-transform duration-200 ${
        isExiting ? 'translate-x-full' : 'translate-x-0'
      } ${colorMap[toast.type]}`}
    >
      <div className="flex-shrink-0 mt-0.5">{iconMap[toast.type]}</div>
      <div className="flex-1 min-w-0">{toast.message}</div>
      <button
        type="button"
        onClick={handleDismiss}
        className="flex-shrink-0 text-xs opacity-60 hover:opacity-100 transition-opacity"
        aria-label="Dismiss"
      >
        <X size={16} />
      </button>
    </div>
  )
}

export const ToastContainer: React.FC = () => {
  const toasts = useToastStore((s) => s.toasts)

  return (
    <div className="fixed bottom-4 right-4 z-[200] space-y-2 pointer-events-none">
      {toasts.map((toast) => (
        <div key={toast.id} className="pointer-events-auto">
          <ToastItem toast={toast} />
        </div>
      ))}
    </div>
  )
}
