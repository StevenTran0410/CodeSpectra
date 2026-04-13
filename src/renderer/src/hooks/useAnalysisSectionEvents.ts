import { useEffect, useState } from 'react'
import type { SectionDoneEvent } from '../types/electron'

type SectionStatus = 'pending' | 'done' | 'error'
export type AnalysisSectionId = 'A' | 'B' | 'C' | 'D' | 'E' | 'F' | 'G' | 'H' | 'I' | 'J' | 'K' | 'L'

const ALL_SECTIONS: AnalysisSectionId[] = [
  'A',
  'B',
  'C',
  'D',
  'E',
  'F',
  'G',
  'H',
  'I',
  'J',
  'K',
  'L',
]

const INITIAL_STATES = (): Record<AnalysisSectionId, SectionStatus> =>
  Object.fromEntries(ALL_SECTIONS.map((s) => [s, 'pending'])) as Record<
    AnalysisSectionId,
    SectionStatus
  >

export interface AnalysisSectionEventsResult {
  sectionStates: Record<AnalysisSectionId, SectionStatus>
  liveSections: Record<AnalysisSectionId, unknown>
}

function applyEvent(
  evt: SectionDoneEvent,
  setStates: React.Dispatch<React.SetStateAction<Record<AnalysisSectionId, SectionStatus>>>,
  setLive: React.Dispatch<React.SetStateAction<Record<AnalysisSectionId, unknown>>>,
) {
  const section = evt.section as AnalysisSectionId
  if (!ALL_SECTIONS.includes(section)) return
  setStates((prev) => ({ ...prev, [section]: evt.status }))
  if (evt.status === 'done') {
    setLive((prev) => ({ ...prev, [section]: evt.data ?? null }))
  }
}

export function useAnalysisSectionEvents(
  activeJobId: string | undefined,
  isRunning: boolean
): AnalysisSectionEventsResult {
  const [sectionStates, setSectionStates] = useState<Record<AnalysisSectionId, SectionStatus>>(
    INITIAL_STATES
  )
  const [liveSections, setLiveSections] = useState<Record<AnalysisSectionId, unknown>>(() =>
    Object.fromEntries(ALL_SECTIONS.map((s) => [s, null])) as Record<AnalysisSectionId, unknown>
  )

  useEffect(() => {
    if (!activeJobId || !isRunning) return

    setSectionStates(INITIAL_STATES())
    setLiveSections(
      Object.fromEntries(ALL_SECTIONS.map((s) => [s, null])) as Record<AnalysisSectionId, unknown>
    )

    // Catch up on events that happened while component was unmounted
    window.api.analysis.pollEvents(activeJobId, 0)
      .then((res) => {
        for (const evt of res.events) {
          applyEvent(evt, setSectionStates, setLiveSections)
        }
      })
      .catch(() => {})

    // Listen for new events going forward
    const handler = (_event: unknown, evt: SectionDoneEvent) => {
      applyEvent(evt, setSectionStates, setLiveSections)
    }

    window.api.analysis.onSectionDone(handler)
    return () => {
      window.api.analysis.offSectionDone(handler)
    }
  }, [activeJobId, isRunning])

  return { sectionStates, liveSections }
}
