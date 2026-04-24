import { useQAStore } from '../store/qa.store'

export interface AskScreenSelectors {
  conversations: ReturnType<typeof useQAStore>['conversations']
  activeId: ReturnType<typeof useQAStore>['activeId']
  createConversation: ReturnType<typeof useQAStore>['createConversation']
  deleteConversation: ReturnType<typeof useQAStore>['deleteConversation']
  renameConversation: ReturnType<typeof useQAStore>['renameConversation']
  updateConversationSnapshot: ReturnType<typeof useQAStore>['updateConversationSnapshot']
  addMessage: ReturnType<typeof useQAStore>['addMessage']
  truncateMessages: ReturnType<typeof useQAStore>['truncateMessages']
  setActive: ReturnType<typeof useQAStore>['setActive']
  disableDeepResearch: ReturnType<typeof useQAStore>['disableDeepResearch']
  isDeepResearchDisabled: ReturnType<typeof useQAStore>['isDeepResearchDisabled']
}

/**
 * Custom hook that provides Zustand selectors for AskScreen component.
 * Each property uses a separate selector to minimize re-renders on store updates.
 */
export function useAskScreen(): AskScreenSelectors {
  const conversations = useQAStore((s) => s.conversations)
  const activeId = useQAStore((s) => s.activeId)
  const createConversation = useQAStore((s) => s.createConversation)
  const deleteConversation = useQAStore((s) => s.deleteConversation)
  const renameConversation = useQAStore((s) => s.renameConversation)
  const updateConversationSnapshot = useQAStore((s) => s.updateConversationSnapshot)
  const addMessage = useQAStore((s) => s.addMessage)
  const truncateMessages = useQAStore((s) => s.truncateMessages)
  const setActive = useQAStore((s) => s.setActive)
  const disableDeepResearch = useQAStore((s) => s.disableDeepResearch)
  const isDeepResearchDisabled = useQAStore((s) => s.isDeepResearchDisabled)

  return {
    conversations,
    activeId,
    createConversation,
    deleteConversation,
    renameConversation,
    updateConversationSnapshot,
    addMessage,
    truncateMessages,
    setActive,
    disableDeepResearch,
    isDeepResearchDisabled,
  }
}
