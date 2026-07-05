export interface RunListItem {
  id: string;
  target_system_id: string;
  eval_plan_id: string | null;
  started_at: string;
  finished_at: string | null;
  status: string;
  map_path: string | null;
  active_defects: string[];
  pass_rate: number;
  judge_cost: number;
}

export interface ComponentAggregate {
  total: number;
  passed: number;
}

export interface SystemMapComponent {
  id: string;
  role: string;
  model: string | null;
  entry_point: string | null;
  constraints: Array<{ name: string; value: any; source: string }>;
  upstream: string[];
  downstream: string[];
}

export interface SystemMap {
  target_system_id: string;
  discrepancies: string[];
  components: SystemMapComponent[];
}

export interface RunDetailResponse {
  id: string;
  target_system_id: string;
  eval_plan_id: string | null;
  started_at: string;
  finished_at: string | null;
  status: string;
  map_path: string | null;
  active_defects: string[];
  system_map: SystemMap;
  component_aggregates: Record<string, ComponentAggregate>;
  overall_pass_rate: number;
  target?: string | null;
  suite_path?: string | null;
  parent_run_id?: string | null;
  model_overrides?: Record<string, string>;
}

export interface EvaluationDetailItem {
  id: string;
  metric_name: string;
  metric_class: string;
  score: number | null;
  passed: boolean | null;
  details: Record<string, any>;
  evaluator: string | null;
  cost_tokens: number | null;
  trace_id: string | null;
  span_id: string | null;
  root_input: string | null;
  final_output: string | null;
  trace_tokens: number | null;
  trace_latency: number | null;
}

export interface TraceSpan {
  id: string;
  trace_id: string;
  parent_span_id: string | null;
  component_id: string | null;
  span_type: string;
  input_json: string | null;
  output_json: string | null;
  model: string | null;
  tokens_in: number | null;
  tokens_out: number | null;
  latency_ms: number | null;
  started_at: string;
  details_json: string;
}

export interface TraceDetailResponse {
  trace: {
    id: string;
    run_id: string;
    dataset_case_id: string | null;
    root_input: string;
    final_output: string | null;
    total_tokens: number;
    total_latency_ms: number;
  } | null;
  spans: TraceSpan[];
}
