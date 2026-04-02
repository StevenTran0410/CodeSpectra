import React from 'react'
import { Cpu } from 'lucide-react'
import { EmptyState } from '../components/ui/EmptyState'

export default function ProvidersSetup(): React.ReactElement {
  return (
    <>
      <div className="screen-header">
        <h1 className="screen-title">Providers</h1>
        <p className="screen-subtitle">Configure local and cloud LLM providers</p>
      </div>
      <div className="h-[calc(100vh-10rem)]">
        <EmptyState
          icon={<Cpu className="w-7 h-7" />}
          title="No providers configured"
          description="Add a local provider (Ollama, LM Studio) or a cloud provider (OpenAI, Anthropic) to start running analysis."
          action={
            <span className="text-xs text-gray-500 bg-surface-overlay border border-surface-border px-3 py-1.5 rounded-md">
              Coming in RPA-021 / RPA-022
            </span>
          }
        />
      </div>
    </>
  )
}
