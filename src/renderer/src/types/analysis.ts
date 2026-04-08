export const ANALYSIS_REPORT_VERSION = 3 as const

export type Confidence = 'high' | 'medium' | 'low'

export type RuntimeType =
  | 'web_app'
  | 'backend_service'
  | 'monolith'
  | 'monorepo'
  | 'library'
  | 'cli'
  | 'worker'
  | 'cron'
  | 'unknown'

export type RiskCategory =
  | 'god_object'
  | 'circular_import'
  | 'todo_hotspot'
  | 'test_gap'
  | 'blast_radius'
  | 'config_risk'
  | 'heavy_branching'
  | 'generated_mixed'

export type FolderRole =
  | 'domain'
  | 'infrastructure'
  | 'delivery'
  | 'shared'
  | 'test'
  | 'generated'
  | 'unknown'

export type ClassVsFunctional = 'class_heavy' | 'functional_heavy' | 'mixed'

export type RuleSeverity = 'strong' | 'suspected' | 'weak'

/** Slot shape used by SectionG (and nested lists). */
export interface SectionRadarSlot {
  file: string
  reason: string
}

export interface GlossaryTerm {
  term: string
  definition: string
  evidence_files: string[]
}

export interface RiskFinding {
  category: RiskCategory | string
  severity: 'high' | 'medium' | 'low'
  title: string
  rationale: string
  evidence: string[]
}

export interface MainServiceItem {
  name: string
  path: string
  role: string
}

export interface FolderItem {
  path: string
  role: FolderRole | string
  description: string
}

export interface ConventionAspect {
  description: string
  evidence_files: string[]
}

export interface ConventionSignal {
  category: string
  description: string
  evidence: string
}

export interface ForbiddenRule {
  rule: string
  inferred_from: string
  severity: RuleSeverity | string
}

export interface RuleViolation {
  rule: string
  file: string
  description: string
  severity?: 'high' | 'medium' | 'low' | string
}

export interface FeatureMapItem {
  name: string
  description: string
  entrypoint: string
  key_files: string[]
  data_path: string
  tests: string[]
  reading_order: string[]
}

export interface OnboardingStep {
  order: number
  file: string
  goal: string
  outcome: string
}

export interface SectionA {
  repo_name: string
  domain: string
  purpose: string
  runtime_type: RuntimeType | string
  tech_stack: string[]
  business_context: string
  confidence: Confidence
  evidence_files: string[]
  blind_spots: string[]
}

export interface SectionB {
  main_layers: string[]
  frameworks: string[]
  entrypoints: string[]
  main_services: MainServiceItem[]
  external_integrations: string[]
  config_sources: string[]
  database_hints: string[]
  confidence: Confidence
  evidence_files: string[]
  blind_spots: string[]
}

export interface SectionC {
  folders: FolderItem[]
  summary: string
  confidence: Confidence
  evidence_files: string[]
  blind_spots: string[]
}

export interface SectionD {
  naming_style: ConventionAspect | string
  error_handling: ConventionAspect | string
  async_style: ConventionAspect | string
  di_style: ConventionAspect | string
  class_vs_functional: ConventionAspect | string
  test_style: ConventionAspect | string
  signals: ConventionSignal[]
  confidence: Confidence
  evidence_files: string[]
  blind_spots: string[]
}

export interface SectionE {
  rules: ForbiddenRule[]
  violations_found: RuleViolation[]
  confidence: Confidence
  evidence_files: string[]
  blind_spots: string[]
}

export interface SectionF {
  features: FeatureMapItem[]
  confidence: Confidence
  evidence_files: string[]
  blind_spots: string[]
}

export interface SectionG {
  entrypoint: SectionRadarSlot
  backbone: SectionRadarSlot
  critical_config: SectionRadarSlot
  highest_centrality: SectionRadarSlot
  most_dangerous_to_touch: SectionRadarSlot
  read_first: SectionRadarSlot
  other_important: SectionRadarSlot[]
  confidence: Confidence
  evidence_files: string[]
  blind_spots: string[]
}

export interface SectionH {
  steps: OnboardingStep[]
  total_estimated_minutes: number
  confidence: Confidence
  evidence_files: string[]
  blind_spots: string[]
}

export interface SectionI {
  terms: GlossaryTerm[]
  confidence: Confidence
  blind_spots: string[]
}

export interface SectionJ {
  findings: RiskFinding[]
  summary: string
  confidence: Confidence
  evidence_files: string[]
  blind_spots: string[]
}

export interface SectionK {
  overall_confidence: Confidence
  section_scores: Partial<Record<SectionId, Confidence | string>>
  weakest_sections: string[]
  coverage_percentage: number
  notes: string
  blind_spots: string[]
}

export interface SectionL {
  executive_summary: string
  architecture_narrative: string
  tech_stack_snapshot: string
  developer_quickstart: string
  conventions_digest: string
  risk_highlights: string
  reading_path: string
  confidence: string
}

export type SectionId = 'A' | 'B' | 'C' | 'D' | 'E' | 'F' | 'G' | 'H' | 'I' | 'J' | 'K' | 'L'

export interface AnalysisReportV2 {
  version: 2
  sections: Partial<Record<SectionId, unknown>>
}

export interface AnalysisReportV3 {
  version: 3
  sections: Partial<Record<SectionId, unknown>> & { L?: SectionL }
}

export type AnalysisReport = AnalysisReportV2 | AnalysisReportV3
