export interface Workspace {
  id: string
  name: string
  created_at: string
  updated_at: string
  settings: Record<string, unknown>
}

export interface CreateWorkspaceInput {
  name: string
}
