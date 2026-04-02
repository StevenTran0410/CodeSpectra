import React, { type ReactNode } from 'react'

interface Props {
  icon: ReactNode
  title: string
  description?: string
  action?: ReactNode
}

export function EmptyState({ icon, title, description, action }: Props): React.ReactElement {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-4 p-8 text-center select-none">
      <div className="w-14 h-14 rounded-2xl bg-surface-overlay border border-surface-border flex items-center justify-center text-gray-500">
        {icon}
      </div>
      <div className="max-w-xs">
        <p className="text-gray-200 font-medium">{title}</p>
        {description && <p className="text-gray-500 text-sm mt-1">{description}</p>}
      </div>
      {action && <div>{action}</div>}
    </div>
  )
}
