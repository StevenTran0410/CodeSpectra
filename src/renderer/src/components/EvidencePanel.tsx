import React from 'react'

type ChunkSource = {
  chunk_id: string
  rel_path: string
  chunk_index: number
  snippet: string
}

export default function EvidencePanel({
  sectionId,
  sources,
  loading,
  onClose,
}: {
  sectionId: string
  sources: ChunkSource[]
  loading: boolean
  onClose: () => void
}): React.ReactElement {
  return (
    <div className="fixed inset-y-0 right-0 w-96 bg-zinc-950 border-l border-zinc-800 flex flex-col z-50 shadow-2xl">
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
        <span className="text-sm font-semibold text-zinc-200">
          Sources — Section {sectionId}
        </span>
        <button
          onClick={onClose}
          className="text-zinc-500 hover:text-zinc-200 text-xs border border-zinc-700 px-2 py-1 rounded"
        >
          ✕
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {loading && <div className="text-xs text-zinc-500">Loading sources...</div>}
        {!loading && sources.length === 0 && (
          <div className="text-xs text-zinc-500">
            No source chunks recorded. Re-run this section to capture evidence.
          </div>
        )}
        {!loading &&
          sources.map((s) => (
            <div key={s.chunk_id} className="rounded border border-zinc-800 bg-zinc-900/60 p-2">
              <div className="text-[10px] font-mono text-indigo-300 truncate mb-1">
                {s.rel_path}
                <span className="ml-2 text-zinc-500">chunk #{s.chunk_index}</span>
              </div>
              <pre className="text-[10px] text-zinc-400 whitespace-pre-wrap break-all leading-relaxed line-clamp-6">
                {s.snippet}
              </pre>
            </div>
          ))}
      </div>
    </div>
  )
}
