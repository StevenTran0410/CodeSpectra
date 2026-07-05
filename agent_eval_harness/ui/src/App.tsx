import React, { useState, useEffect } from 'react';
import {
  Play,
  CheckCircle2,
  XCircle,
  Clock,
  Compass,
  Layers,
  ArrowRight,
  TrendingDown,
  TrendingUp,
  AlertTriangle,
  Info,
  DollarSign,
  Maximize2,
  GitCommit,
  GitBranch,
  RefreshCw,
  Search,
  Code,
  FileText,
  Boxes,
  Database
} from 'lucide-react';
import {
  RunListItem,
  RunDetailResponse,
  EvaluationDetailItem,
  TraceDetailResponse,
  TraceSpan
} from './types';

// Helper to format timestamps
function formatTime(isoStr: string) {
  try {
    const d = new Date(isoStr);
    return d.toLocaleString();
  } catch (e) {
    return isoStr;
  }
}

// Helper to format latency
function formatLatency(ms: number | null) {
  if (ms === null) return 'N/A';
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export default function App() {
  // Hash Routing State
  const [route, setRoute] = useState<string>('runs');
  const [params, setParams] = useState<Record<string, string>>({});

  // Data State
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [selectedRun, setSelectedRun] = useState<RunDetailResponse | null>(null);
  const [evaluations, setEvaluations] = useState<EvaluationDetailItem[]>([]);
  const [traceData, setTraceData] = useState<TraceDetailResponse | null>(null);
  const [selectedSpan, setSelectedSpan] = useState<TraceSpan | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Comparison State
  const [compareRunIdA, setCompareRunIdA] = useState<string>('');
  const [compareRunIdB, setCompareRunIdB] = useState<string>('');
  const [compareRunA, setCompareRunA] = useState<RunDetailResponse | null>(null);
  const [compareRunB, setCompareRunB] = useState<RunDetailResponse | null>(null);
  const [compareEvalsA, setCompareEvalsA] = useState<EvaluationDetailItem[]>([]);
  const [compareEvalsB, setCompareEvalsB] = useState<EvaluationDetailItem[]>([]);

  // Parser Hash on load and change
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.slice(1) || 'runs';
      const parts = hash.split('/');
      
      if (parts[0] === 'runs') {
        setRoute('runs');
        setParams({});
      } else if (parts[0] === 'run' && parts[1]) {
        if (parts[2] === 'component' && parts[3] && parts[4] === 'trace' && parts[5]) {
          // Trace opened from a component screen — componentId kept in the hash so
          // Back can return to that component instead of jumping to the dashboard
          setRoute('trace');
          setParams({ runId: parts[1], componentId: parts[3], traceId: parts[5], spanId: parts[7] || '' });
        } else if (parts[2] === 'component' && parts[3]) {
          setRoute('component');
          setParams({ runId: parts[1], componentId: parts[3] });
        } else if (parts[2] === 'trace' && parts[3]) {
          setRoute('trace');
          setParams({ runId: parts[1], traceId: parts[3], spanId: parts[5] || '' });
        } else {
          setRoute('run');
          setParams({ runId: parts[1] });
        }
      } else if (parts[0] === 'compare' && parts[1] && parts[2]) {
        setRoute('compare');
        setParams({ runIdA: parts[1], runIdB: parts[2] });
      } else {
        setRoute('runs');
        setParams({});
      }
    };

    window.addEventListener('hashchange', handleHashChange);
    handleHashChange(); // Initial run

    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  // Fetch runs list
  const fetchRunsList = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/runs');
      if (!res.ok) throw new Error('Failed to fetch runs');
      const data = await res.json();
      setRuns(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (route === 'runs') {
      fetchRunsList();
    }
  }, [route]);

  // Fetch Run Detail (Overview / Topology)
  useEffect(() => {
    if (route === 'run' && params.runId) {
      const fetchRunDetail = async () => {
        setLoading(true);
        setError(null);
        try {
          const res = await fetch(`/api/runs/${params.runId}`);
          if (!res.ok) throw new Error('Failed to fetch run details');
          const data = await res.json();
          setSelectedRun(data);
        } catch (e: any) {
          setError(e.message);
        } finally {
          setLoading(false);
        }
      };
      fetchRunDetail();
    }
  }, [route, params.runId]);

  // Fetch Component Evaluations
  useEffect(() => {
    if (route === 'component' && params.runId && params.componentId) {
      const fetchComponentData = async () => {
        setLoading(true);
        setError(null);
        try {
          // Fetch Run meta for breadcrumbs
          const runRes = await fetch(`/api/runs/${params.runId}`);
          if (runRes.ok) {
            const runData = await runRes.json();
            setSelectedRun(runData);
          }
          
          const evalRes = await fetch(`/api/runs/${params.runId}/components/${params.componentId}`);
          if (!evalRes.ok) throw new Error('Failed to fetch component evaluations');
          const evalData = await evalRes.json();
          setEvaluations(evalData);
        } catch (e: any) {
          setError(e.message);
        } finally {
          setLoading(false);
        }
      };
      fetchComponentData();
    }
  }, [route, params.runId, params.componentId]);

  // Fetch Trace Span Tree
  useEffect(() => {
    if (route === 'trace' && params.traceId) {
      const fetchTraceData = async () => {
        setLoading(true);
        setError(null);
        try {
          const res = await fetch(`/api/traces/${params.traceId}`);
          if (!res.ok) throw new Error('Failed to fetch trace spans');
          const data = await res.json();
          setTraceData(data);
          
          // Select default span if not set
          if (params.spanId) {
            const span = data.spans.find((s: TraceSpan) => s.id === params.spanId);
            if (span) setSelectedSpan(span);
          } else if (data.spans.length > 0) {
            setSelectedSpan(data.spans[0]);
          }
        } catch (e: any) {
          setError(e.message);
        } finally {
          setLoading(false);
        }
      };
      fetchTraceData();
    }
  }, [route, params.traceId, params.spanId]);

  // Fetch Comparison Data
  useEffect(() => {
    if (route === 'compare' && params.runIdA && params.runIdB) {
      const fetchComparison = async () => {
        setLoading(true);
        setError(null);
        try {
          const [resA, resB] = await Promise.all([
            fetch(`/api/runs/${params.runIdA}`),
            fetch(`/api/runs/${params.runIdB}`)
          ]);
          if (!resA.ok || !resB.ok) throw new Error('Failed to fetch compared run details');
          const dataA = await resA.json();
          const dataB = await resB.json();
          setSelectedRun(dataA); // Keep run A as base
          setCompareRunA(dataA);
          setCompareRunB(dataB);

          // Get evaluations for all components to perform diffing
          const compsA = Object.keys(dataA.component_aggregates);
          const evalsPromisesA = compsA.map(cid =>
            fetch(`/api/runs/${params.runIdA}/components/${cid}`).then(r => r.json())
          );
          const evalsPromisesB = compsA.map(cid =>
            fetch(`/api/runs/${params.runIdB}/components/${cid}`).then(r => r.json())
          );

          const evalsA = await Promise.all(evalsPromisesA);
          const evalsB = await Promise.all(evalsPromisesB);

          setCompareEvalsA(evalsA.flat());
          setCompareEvalsB(evalsB.flat());
        } catch (e: any) {
          setError(e.message);
        } finally {
          setLoading(false);
        }
      };
      fetchComparison();
    }
  }, [route, params.runIdA, params.runIdB]);

  // Navigation helpers
  const navigateToRuns = () => {
    window.location.hash = 'runs';
  };

  const navigateToRun = (runId: string) => {
    window.location.hash = `run/${runId}`;
  };

  const navigateToComponent = (runId: string, componentId: string) => {
    window.location.hash = `run/${runId}/component/${componentId}`;
  };

  const navigateToTrace = (runId: string, traceId: string, spanId?: string, componentId?: string) => {
    const base = componentId ? `run/${runId}/component/${componentId}` : `run/${runId}`;
    window.location.hash = `${base}/trace/${traceId}${spanId ? `/span/${spanId}` : ''}`;
  };

  // Hierarchical back: one level up the drill-down path, never straight to the
  // dashboard (that jump was the only affordance before and lost the user's place).
  const navigateBack = () => {
    if (route === 'component') navigateToRun(params.runId);
    else if (route === 'trace' && params.componentId)
      navigateToComponent(params.runId, params.componentId);
    else if (route === 'trace') navigateToRun(params.runId);
    else navigateToRuns();
  };
  const backLabel =
    route === 'component'
      ? 'Back to Run Details'
      : route === 'trace'
      ? params.componentId
        ? `Back to ${params.componentId}`
        : 'Back to Run Details'
      : 'Back to Runs';

  const navigateToCompare = (runIdA: string, runIdB: string) => {
    window.location.hash = `compare/${runIdA}/${runIdB}`;
  };

  return (
    <div className="min-h-screen bg-[#070b12] text-slate-100 flex flex-col selection:bg-indigo-500 selection:text-white">
      {/* Premium Header */}
      <header className="border-b border-slate-800/80 bg-slate-950/60 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3 cursor-pointer" onClick={navigateToRuns}>
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Layers className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
              AEH Dashboard
            </h1>
            <p className="text-xs text-slate-400 font-medium">Agentic Evaluation Harness</p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          {route !== 'runs' && (
            <button
              onClick={navigateBack}
              className="px-4 py-2 text-sm font-medium text-slate-400 hover:text-slate-200 bg-slate-900 border border-slate-800 hover:border-slate-700 rounded-lg transition-all"
            >
              ← {backLabel}
            </button>
          )}
          <button
            onClick={() => {
              if (route === 'runs') fetchRunsList();
              else if (route === 'run') navigateToRun(params.runId);
            }}
            className="p-2 text-slate-400 hover:text-indigo-400 hover:bg-slate-900 border border-slate-800 hover:border-indigo-500/30 rounded-lg transition-all"
            title="Refresh current view"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 flex flex-col">
        {error && (
          <div className="mb-6 p-4 rounded-xl border border-red-500/20 bg-red-950/20 text-red-400 flex items-center space-x-3">
            <AlertTriangle className="h-5 w-5 flex-shrink-0" />
            <p className="text-sm font-medium">{error}</p>
          </div>
        )}

        {loading && (
          <div className="flex-1 flex flex-col items-center justify-center py-20 space-y-4">
            <div className="relative">
              <div className="h-12 w-12 rounded-full border-t-2 border-indigo-500 animate-spin"></div>
            </div>
            <p className="text-sm text-slate-400 font-medium">Loading store details...</p>
          </div>
        )}

        {!loading && (
          <>
            {/* SCREEN 1: RUNS LIST */}
            {route === 'runs' && (
              <div className="space-y-6">
                {/* Dashboard Stats */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="glass rounded-2xl p-6 relative overflow-hidden">
                    <div className="absolute top-0 right-0 h-32 w-32 bg-indigo-500/5 rounded-full blur-3xl -mr-8 -mt-8"></div>
                    <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Total Runs</p>
                    <p className="text-4xl font-extrabold text-white mt-2">{runs.length}</p>
                    <div className="flex items-center space-x-2 mt-4 text-xs text-indigo-400 font-medium">
                      <Database className="h-4 w-4" />
                      <span>Loaded from SQLite store</span>
                    </div>
                  </div>
                  <div className="glass rounded-2xl p-6 relative overflow-hidden">
                    <div className="absolute top-0 right-0 h-32 w-32 bg-emerald-500/5 rounded-full blur-3xl -mr-8 -mt-8"></div>
                    <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Latest Run Pass Rate</p>
                    <p className="text-4xl font-extrabold text-emerald-400 mt-2">
                      {runs.length > 0
                        ? `${(runs[0].pass_rate * 100).toFixed(0)}%`
                        : 'N/A'}
                    </p>
                    <div className="flex items-center space-x-2 mt-4 text-xs text-emerald-500 font-medium">
                      <CheckCircle2 className="h-4 w-4" />
                      <span>Metric-level correctness</span>
                    </div>
                  </div>
                  <div className="glass rounded-2xl p-6 relative overflow-hidden">
                    <div className="absolute top-0 right-0 h-32 w-32 bg-purple-500/5 rounded-full blur-3xl -mr-8 -mt-8"></div>
                    <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Run Comparison</p>
                    <div className="flex items-center space-x-2 mt-2">
                      <select
                        value={compareRunIdA}
                        onChange={(e) => setCompareRunIdA(e.target.value)}
                        className="bg-slate-900 border border-slate-800 text-xs rounded-lg p-2 text-slate-300 w-full focus:outline-none focus:border-indigo-500"
                      >
                        <option value="">Baseline Run</option>
                        {runs.map((r) => (
                          <option key={r.id} value={r.id}>
                            {r.target_system_id} ({r.id.slice(0, 8)})
                          </option>
                        ))}
                      </select>
                      <ArrowRight className="h-4 w-4 text-slate-500 flex-shrink-0" />
                      <select
                        value={compareRunIdB}
                        onChange={(e) => setCompareRunIdB(e.target.value)}
                        className="bg-slate-900 border border-slate-800 text-xs rounded-lg p-2 text-slate-300 w-full focus:outline-none focus:border-indigo-500"
                      >
                        <option value="">Comparison Run</option>
                        {runs.map((r) => (
                          <option key={r.id} value={r.id}>
                            {r.target_system_id} ({r.id.slice(0, 8)})
                          </option>
                        ))}
                      </select>
                    </div>
                    <button
                      onClick={() => navigateToCompare(compareRunIdA, compareRunIdB)}
                      disabled={!compareRunIdA || !compareRunIdB}
                      className="mt-3 w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 disabled:from-slate-800 disabled:to-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed font-medium text-xs text-white py-2 px-3 rounded-lg shadow-md transition-all flex items-center justify-center space-x-2"
                    >
                      <TrendingUp className="h-3.5 w-3.5" />
                      <span>Compare Selected Runs</span>
                    </button>
                  </div>
                </div>

                {/* Runs Table */}
                <div className="glass rounded-2xl overflow-hidden">
                  <div className="px-6 py-4 border-b border-slate-800/80 bg-slate-900/40 flex items-center justify-between">
                    <h3 className="text-md font-bold text-white flex items-center space-x-2">
                      <Boxes className="h-4 w-4 text-indigo-400" />
                      <span>Evaluation Runs</span>
                    </h3>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="border-b border-slate-800/60 bg-slate-950/20 text-slate-400 text-xs font-semibold uppercase tracking-wider">
                          <th className="py-4 px-6">Target System</th>
                          <th className="py-4 px-6">Run ID / Plan</th>
                          <th className="py-4 px-6">Timestamp</th>
                          <th className="py-4 px-6">Defects Active</th>
                          <th className="py-4 px-6">Pass Rate</th>
                          <th className="py-4 px-6">Cost</th>
                          <th className="py-4 px-6">Status</th>
                          <th className="py-4 px-6 text-right">Actions</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/40 text-sm">
                        {runs.length === 0 ? (
                          <tr>
                            <td colSpan={8} className="py-12 text-center text-slate-500 font-medium">
                              No evaluation runs found in database. Run `aeh eval` first to populate.
                            </td>
                          </tr>
                        ) : (
                          runs.map((run) => (
                            <tr key={run.id} className="hover:bg-slate-900/30 transition-colors group">
                              <td className="py-4 px-6 font-bold text-white">
                                {run.target_system_id}
                              </td>
                              <td className="py-4 px-6">
                                <span className="font-mono text-xs text-slate-400 block">
                                  {run.id.slice(0, 8)}...
                                </span>
                                <span className="text-xs text-slate-500 block truncate max-w-xs">
                                  {run.eval_plan_id || 'no suite plan'}
                                </span>
                              </td>
                              <td className="py-4 px-6 text-xs text-slate-400">
                                {formatTime(run.started_at)}
                              </td>
                              <td className="py-4 px-6">
                                <div className="flex flex-wrap gap-1.5 max-w-xs">
                                  {run.active_defects.length === 0 ? (
                                    <span className="text-xs text-slate-500 italic">None (Clean)</span>
                                  ) : (
                                    run.active_defects.map((def) => (
                                      <span
                                        key={def}
                                        className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-red-950/30 text-red-400 border border-red-900/40"
                                      >
                                        {def.replace('DEFECT_', '')}
                                      </span>
                                    ))
                                  )}
                                </div>
                              </td>
                              <td className="py-4 px-6">
                                <div className="flex items-center space-x-2">
                                  <div className="w-16 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                                    <div
                                      className={`h-full rounded-full ${
                                        run.pass_rate === 1.0 ? 'bg-emerald-500' : 'bg-indigo-500'
                                      }`}
                                      style={{ width: `${run.pass_rate * 100}%` }}
                                    ></div>
                                  </div>
                                  <span className="text-xs font-bold text-white">
                                    {(run.pass_rate * 100).toFixed(0)}%
                                  </span>
                                </div>
                              </td>
                              <td className="py-4 px-6 text-xs text-purple-400 font-bold">
                                {run.judge_cost > 0 ? (
                                  <span className="flex items-center">
                                    <DollarSign className="h-3 w-3 mr-0.5" />
                                    {run.judge_cost} tokens
                                  </span>
                                ) : (
                                  <span className="text-slate-500">Free</span>
                                )}
                              </td>
                              <td className="py-4 px-6">
                                <span
                                  className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${
                                    run.status === 'completed'
                                      ? 'bg-emerald-950/20 text-emerald-400 border-emerald-900/30'
                                      : run.status === 'failed'
                                      ? 'bg-red-950/20 text-red-400 border-red-900/30'
                                      : 'bg-indigo-950/20 text-indigo-400 border-indigo-900/30 animate-pulse'
                                  }`}
                                >
                                  {run.status}
                                </span>
                              </td>
                              <td className="py-4 px-6 text-right">
                                <button
                                  onClick={() => navigateToRun(run.id)}
                                  className="inline-flex items-center space-x-1 text-xs font-bold text-indigo-400 hover:text-indigo-300 group-hover:translate-x-1 transition-all"
                                >
                                  <span>View Report</span>
                                  <ArrowRight className="h-3.5 w-3.5" />
                                </button>
                              </td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* SCREEN 2: RUN OVERVIEW / TOPOLOGY */}
            {route === 'run' && selectedRun && (
              <div className="space-y-6">
                {/* Breadcrumbs */}
                <div className="flex items-center space-x-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  <span className="hover:text-white cursor-pointer" onClick={navigateToRuns}>
                    Runs
                  </span>
                  <span>/</span>
                  <span className="text-white">Run Details ({selectedRun.id.slice(0, 8)})</span>
                </div>

                {/* Meta details */}
                <div className="glass rounded-2xl p-6 grid grid-cols-1 md:grid-cols-4 gap-6">
                  <div>
                    <p className="text-xs text-slate-400 font-semibold uppercase">Target System</p>
                    <p className="text-lg font-bold text-white mt-1">{selectedRun.target_system_id}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-400 font-semibold uppercase">Plan File</p>
                    <p className="text-sm font-mono text-slate-300 mt-1 truncate" title={selectedRun.map_path || ''}>
                      {selectedRun.eval_plan_id || 'Direct Run (No Plan)'}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-400 font-semibold uppercase">Timestamp</p>
                    <p className="text-sm text-slate-300 mt-1">{formatTime(selectedRun.started_at)}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-400 font-semibold uppercase">Status</p>
                    <p className="mt-1">
                      <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-950/20 text-emerald-400 border border-emerald-900/30">
                        {selectedRun.status}
                      </span>
                    </p>
                  </div>
                </div>

                {/* Topology & Components Overview */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Left Column: Topology Map (Screen 2 Topology Graph) */}
                  <div className="glass rounded-2xl p-6 lg:col-span-2 space-y-4">
                    <h3 className="text-md font-bold text-white flex items-center space-x-2">
                      <Compass className="h-5 w-5 text-indigo-400" />
                      <span>System Topology Graph</span>
                    </h3>
                    <p className="text-xs text-slate-400">
                      Execution flow mapping component dependencies and aggregated evaluation pass rates.
                    </p>

                    <div className="border border-slate-800/80 rounded-xl bg-slate-950/50 p-6 min-h-[400px] flex flex-col justify-between">
                      {selectedRun.system_map.components.length === 0 ? (
                        <div className="flex-1 flex items-center justify-center text-slate-500 text-xs italic">
                          No topology map layout loaded.
                        </div>
                      ) : (
                        <div className="space-y-6">
                          {/* We draw components ordered from Upstream to Downstream */}
                          {selectedRun.system_map.components.map((comp) => {
                            const score = selectedRun.component_aggregates[comp.id];
                            const pct = score ? (score.passed / score.total) * 100 : null;

                            return (
                              <div
                                key={comp.id}
                                onClick={() => navigateToComponent(selectedRun.id, comp.id)}
                                className="glass glass-hover p-4 rounded-xl cursor-pointer flex items-center justify-between border-slate-800 hover:border-indigo-500/40"
                              >
                                <div className="flex items-center space-x-3">
                                  <div className={`h-2 w-2 rounded-full ${
                                    pct === 100 ? 'bg-emerald-500' : pct !== null && pct < 100 ? 'bg-amber-500' : 'bg-slate-600'
                                  }`} />
                                  <div>
                                    <p className="text-sm font-bold text-white">{comp.id}</p>
                                    <p className="text-[10px] text-slate-500 font-semibold uppercase">{comp.role}</p>
                                  </div>
                                </div>

                                <div className="flex items-center space-x-4">
                                  {comp.model && (
                                    <span className="text-[10px] font-mono bg-slate-900 border border-slate-800 text-slate-400 px-1.5 py-0.5 rounded">
                                      {comp.model}
                                    </span>
                                  )}

                                  {score ? (
                                    <div className="text-right">
                                      <p className="text-xs font-bold text-white">
                                        {score.passed}/{score.total} passed
                                      </p>
                                      <p className={`text-[10px] font-bold ${
                                        pct === 100 ? 'text-emerald-400' : 'text-amber-400'
                                      }`}>
                                        {pct?.toFixed(0)}% Correct
                                      </p>
                                    </div>
                                  ) : (
                                    <span className="text-xs text-slate-500 italic">No metrics</span>
                                  )}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Right Column: Active Defects & Configs */}
                  <div className="glass rounded-2xl p-6 space-y-4">
                    <h3 className="text-md font-bold text-white flex items-center space-x-2">
                      <AlertTriangle className="h-5 w-5 text-indigo-400" />
                      <span>Planted Defects Status</span>
                    </h3>
                    <p className="text-xs text-slate-400">
                      Currently active switches from the regression gauntlet.
                    </p>

                    <div className="space-y-3">
                      {[
                        // Matches DefectConfig.active_names() (test_targets/_shared/defects.py),
                        // which returns lowercase dataclass field names with no "DEFECT_" prefix
                        // (e.g. "writer_hallucinate") — comparing against the uppercase/prefixed
                        // form here always returned false, so every switch showed OFF regardless
                        // of what was actually active.
                        'planner_overpack',
                        'guard_leak',
                        'wrong_tool',
                        'judge_rubber_stamp',
                        'writer_hallucinate',
                        'no_retry',
                      ].map((def) => {
                        const isActive = selectedRun.active_defects.includes(def);
                        return (
                          <div
                            key={def}
                            className={`p-3 rounded-lg border flex items-center justify-between text-xs ${
                              isActive
                                ? 'bg-red-950/20 border-red-900/30 text-red-400 font-bold'
                                : 'bg-slate-900/40 border-slate-800 text-slate-500'
                            }`}
                          >
                            <span>{def.toUpperCase()}</span>
                            <span className="uppercase text-[10px]">
                              {isActive ? 'Active' : 'Off'}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* SCREEN 3: COMPONENT DETAIL */}
            {route === 'component' && selectedRun && (
              <div className="space-y-6">
                {/* Breadcrumbs */}
                <div className="flex items-center space-x-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  <span className="hover:text-white cursor-pointer" onClick={navigateToRuns}>
                    Runs
                  </span>
                  <span>/</span>
                  <span className="hover:text-white cursor-pointer" onClick={() => navigateToRun(selectedRun.id)}>
                    Run Details ({selectedRun.id.slice(0, 8)})
                  </span>
                  <span>/</span>
                  <span className="text-white">Component: {params.componentId}</span>
                </div>

                {/* Component Detail Summary card */}
                <div className="glass rounded-2xl p-6 relative overflow-hidden">
                  <h2 className="text-xl font-bold text-white">{params.componentId}</h2>
                  <p className="text-xs text-slate-400 mt-1 uppercase tracking-wider font-semibold">
                    Component Configuration details from system map
                  </p>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
                    <div className="bg-slate-950/40 border border-slate-800/80 rounded-xl p-4">
                      <p className="text-[10px] text-slate-500 font-bold uppercase">Component Role</p>
                      <p className="text-sm font-bold text-slate-300 mt-1">
                        {selectedRun.system_map.components.find((c) => c.id === params.componentId)?.role || 'N/A'}
                      </p>
                    </div>
                    <div className="bg-slate-950/40 border border-slate-800/80 rounded-xl p-4">
                      <p className="text-[10px] text-slate-500 font-bold uppercase">Implementation Model</p>
                      <p className="text-sm font-bold text-slate-300 mt-1 font-mono">
                        {selectedRun.system_map.components.find((c) => c.id === params.componentId)?.model || 'N/A'}
                      </p>
                    </div>
                    <div className="bg-slate-950/40 border border-slate-800/80 rounded-xl p-4">
                      <p className="text-[10px] text-slate-500 font-bold uppercase">Entry Point</p>
                      <p className="text-sm font-bold text-slate-300 mt-1 font-mono truncate" title={selectedRun.system_map.components.find((c) => c.id === params.componentId)?.entry_point || ''}>
                        {selectedRun.system_map.components.find((c) => c.id === params.componentId)?.entry_point || 'N/A'}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Drill Down Panels grouped by metric_class (Screen 3 acceptance) */}
                <div className="space-y-6">
                  {/* Class 1: Assertions */}
                  {evaluations.filter((ev) => ev.metric_class === 'assertion').length > 0 && (
                    <div className="glass rounded-2xl overflow-hidden">
                      <div className="px-6 py-4 border-b border-slate-800/80 bg-slate-900/40 flex items-center justify-between">
                        <h3 className="text-md font-bold text-white flex items-center space-x-2">
                          <Code className="h-4 w-4 text-indigo-400" />
                          <span>Structural Assertions</span>
                        </h3>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-left border-collapse">
                          <thead>
                            <tr className="border-b border-slate-800/60 bg-slate-950/20 text-slate-400 text-xs font-semibold uppercase">
                              <th className="py-3 px-6">Assertion Metric</th>
                              <th className="py-3 px-6">User Query Context</th>
                              <th className="py-3 px-6">Details / Params</th>
                              <th className="py-3 px-6">Outcome</th>
                              <th className="py-3 px-6 text-right">Trace Link</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-800/40 text-sm">
                            {evaluations
                              .filter((ev) => ev.metric_class === 'assertion')
                              .map((ev) => (
                                <tr key={ev.id} className="hover:bg-slate-900/10">
                                  <td className="py-4 px-6 font-semibold text-white">
                                    {ev.metric_name}
                                  </td>
                                  <td className="py-4 px-6 text-xs text-slate-300 max-w-xs truncate" title={ev.root_input || ''}>
                                    {ev.root_input || 'N/A'}
                                  </td>
                                  <td className="py-4 px-6 text-xs font-mono text-slate-400 max-w-xs truncate" title={JSON.stringify(ev.details)}>
                                    {JSON.stringify(ev.details)}
                                  </td>
                                  <td className="py-4 px-6">
                                    <span className={`inline-flex items-center space-x-1 font-semibold text-xs ${
                                      ev.passed ? 'text-emerald-400' : 'text-red-400'
                                    }`}>
                                      {ev.passed ? <CheckCircle2 className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
                                      <span>{ev.passed ? 'PASS' : 'FAIL'}</span>
                                    </span>
                                  </td>
                                  <td className="py-4 px-6 text-right">
                                    {ev.trace_id && (
                                      <button
                                        onClick={() => navigateToTrace(selectedRun.id, ev.trace_id || '', ev.span_id || '', params.componentId)}
                                        className="text-xs text-indigo-400 hover:text-indigo-300 font-bold"
                                      >
                                        Open Trace Tree
                                      </button>
                                    )}
                                  </td>
                                </tr>
                              ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Class 2: Classifiers (Confusion Matrix) */}
                  {evaluations.filter((ev) => ev.metric_class === 'classifier').length > 0 && (
                    <div className="glass rounded-2xl overflow-hidden p-6 space-y-4">
                      <h3 className="text-md font-bold text-white flex items-center space-x-2 border-b border-slate-800 pb-3">
                        <GitBranch className="h-4 w-4 text-indigo-400" />
                        <span>Classification Accuracy (Confusion Matrix)</span>
                      </h3>

                      {evaluations
                        .filter((ev) => ev.metric_class === 'classifier')
                        .map((ev) => {
                          const matrix = ev.details.confusion_matrix || ev.details.matrix || {};
                          return (
                            <div key={ev.id} className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                              <div>
                                <p className="text-sm font-bold text-indigo-400">{ev.metric_name}</p>
                                <p className="text-xs text-slate-400 mt-2 font-medium">
                                  Overall Accuracy Score: <span className="font-bold text-white">{ev.score !== null ? `${(ev.score * 100).toFixed(1)}%` : 'N/A'}</span>
                                </p>
                                {ev.details.unmatched_classes && (
                                  <p className="text-xs text-amber-500 mt-1 font-semibold">
                                    Unmatched prediction tags occurred.
                                  </p>
                                )}
                              </div>

                              <div className="border border-slate-800 rounded-xl p-4 bg-slate-950/60 font-mono text-xs text-slate-300">
                                <p className="text-[10px] font-bold text-slate-500 uppercase border-b border-slate-800 pb-2 mb-2">
                                  Confusion Matrix Grid
                                </p>
                                <pre className="overflow-x-auto">
                                  {JSON.stringify(matrix, null, 2)}
                                </pre>
                              </div>
                            </div>
                          );
                        })}
                    </div>
                  )}

                  {/* Class 3: LLM Judges (G-Eval / Ragas distributions) */}
                  {evaluations.filter((ev) => ev.metric_class === 'llm_judge').length > 0 && (
                    <div className="glass rounded-2xl overflow-hidden p-6 space-y-6">
                      <h3 className="text-md font-bold text-white flex items-center space-x-2 border-b border-slate-800 pb-3">
                        <FileText className="h-4 w-4 text-indigo-400" />
                        <span>LLM Judge Evaluations</span>
                      </h3>

                      <div className="space-y-6">
                        {evaluations
                          .filter((ev) => ev.metric_class === 'llm_judge')
                          .map((ev) => (
                            <div key={ev.id} className="border border-slate-800/80 rounded-xl p-5 bg-slate-950/30 space-y-3">
                              <div className="flex items-center justify-between">
                                <div>
                                  <p className="text-sm font-bold text-white">{ev.metric_name}</p>
                                  <p className="text-[10px] text-slate-500 font-semibold uppercase">Evaluated via: {ev.evaluator || 'LLM'}</p>
                                </div>

                                <div className="text-right">
                                  <span className="text-lg font-bold text-indigo-400">
                                    {ev.score !== null ? ev.score.toFixed(2) : 'N/A'}
                                  </span>
                                  {ev.cost_tokens && (
                                    <p className="text-[10px] text-purple-400 font-bold">
                                      {ev.cost_tokens} tokens
                                    </p>
                                  )}
                                </div>
                              </div>

                              {ev.details.rubric_text && (
                                <div className="p-3 bg-slate-900 border border-slate-800/60 rounded-lg text-xs text-slate-400 font-medium">
                                  <span className="font-bold text-slate-300 block mb-1">Tailored Rubric:</span>
                                  {ev.details.rubric_text}
                                </div>
                              )}

                              {ev.details.reason && (
                                <div className="text-xs text-slate-300 italic pl-3 border-l-2 border-slate-700">
                                  &ldquo;{ev.details.reason}&rdquo;
                                </div>
                              )}

                              <div className="flex items-center justify-between text-xs mt-3 border-t border-slate-800/40 pt-3">
                                <span className="text-slate-500">Query ID: {ev.trace_id?.slice(0, 8)}...</span>
                                {ev.trace_id && (
                                  <button
                                    onClick={() => navigateToTrace(selectedRun.id, ev.trace_id || '', ev.span_id || '', params.componentId)}
                                    className="text-indigo-400 hover:text-indigo-300 font-bold"
                                  >
                                    View Spans In Trace Tree &rarr;
                                  </button>
                                )}
                              </div>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* SCREEN 4: TRACE VIEWER */}
            {route === 'trace' && traceData && (
              <div className="space-y-6 flex-1 flex flex-col">
                {/* Breadcrumbs */}
                <div className="flex items-center space-x-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  <span className="hover:text-white cursor-pointer" onClick={navigateToRuns}>
                    Runs
                  </span>
                  <span>/</span>
                  <span className="hover:text-white cursor-pointer" onClick={() => navigateToRun(params.runId)}>
                    Run Details ({params.runId.slice(0, 8)})
                  </span>
                  <span>/</span>
                  {params.componentId && (
                    <>
                      <span
                        className="hover:text-white cursor-pointer"
                        onClick={() => navigateToComponent(params.runId, params.componentId)}
                      >
                        Component: {params.componentId}
                      </span>
                      <span>/</span>
                    </>
                  )}
                  <span className="text-white">Trace Explorer: {params.traceId.slice(0, 8)}...</span>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 items-stretch">
                  {/* Left Column: Trace Span Tree */}
                  <div className="glass rounded-2xl p-6 lg:col-span-1 space-y-4 flex flex-col">
                    <h3 className="text-md font-bold text-white flex items-center space-x-2 border-b border-slate-800 pb-3">
                      <GitCommit className="h-5 w-5 text-indigo-400" />
                      <span>Trace Execution Spans</span>
                    </h3>

                    <div className="space-y-2 flex-1 overflow-y-auto max-h-[500px] pr-2">
                      {traceData.spans.map((span) => {
                        const isSelected = selectedSpan?.id === span.id;
                        return (
                          <div
                            key={span.id}
                            onClick={() => setSelectedSpan(span)}
                            className={`p-3 rounded-lg border text-left cursor-pointer transition-all ${
                              isSelected
                                ? 'bg-indigo-950/20 border-indigo-500 text-indigo-300 font-bold'
                                : 'bg-slate-900/40 border-slate-800/80 text-slate-400 hover:border-slate-700'
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-bold truncate max-w-[120px]">{span.component_id || span.span_type}</span>
                              <span className="text-[10px] px-1 bg-slate-800 text-slate-500 rounded font-mono">
                                {span.span_type}
                              </span>
                            </div>
                            <div className="flex items-center justify-between text-[10px] text-slate-500 mt-2">
                              <span>{formatLatency(span.latency_ms)}</span>
                              {span.tokens_in !== null && (
                                <span>{span.tokens_in + (span.tokens_out || 0)} tokens</span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Right Column: Span detail pane showing input/output */}
                  <div className="glass rounded-2xl p-6 lg:col-span-2 space-y-4 flex flex-col">
                    <h3 className="text-md font-bold text-white flex items-center space-x-2 border-b border-slate-800 pb-3">
                      <Code className="h-5 w-5 text-indigo-400" />
                      <span>Span Details: {selectedSpan?.component_id || selectedSpan?.span_type || 'N/A'}</span>
                    </h3>

                    {selectedSpan ? (
                      <div className="space-y-4 flex-1 flex flex-col">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                          <div className="bg-slate-900/60 p-3 border border-slate-800/80 rounded-xl">
                            <span className="text-[10px] text-slate-500 font-bold uppercase">Span ID</span>
                            <span className="font-mono text-slate-300 block mt-1">{selectedSpan.id.slice(0, 8)}...</span>
                          </div>
                          <div className="bg-slate-900/60 p-3 border border-slate-800/80 rounded-xl">
                            <span className="text-[10px] text-slate-500 font-bold uppercase">Latency</span>
                            <span className="text-slate-300 block mt-1 font-semibold">{formatLatency(selectedSpan.latency_ms)}</span>
                          </div>
                          <div className="bg-slate-900/60 p-3 border border-slate-800/80 rounded-xl">
                            <span className="text-[10px] text-slate-500 font-bold uppercase">Tokens In</span>
                            <span className="text-slate-300 block mt-1 font-semibold">{selectedSpan.tokens_in ?? 'N/A'}</span>
                          </div>
                          <div className="bg-slate-900/60 p-3 border border-slate-800/80 rounded-xl">
                            <span className="text-[10px] text-slate-500 font-bold uppercase">Tokens Out</span>
                            <span className="text-slate-300 block mt-1 font-semibold">{selectedSpan.tokens_out ?? 'N/A'}</span>
                          </div>
                        </div>

                        {/* Input JSON */}
                        <div className="space-y-1.5">
                          <p className="text-[10px] text-slate-500 font-bold uppercase">Received Payload / Input</p>
                          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 font-mono text-xs text-slate-300 overflow-x-auto max-h-[160px]">
                            <pre>{JSON.stringify(JSON.parse(selectedSpan.input_json || '{}'), null, 2)}</pre>
                          </div>
                        </div>

                        {/* Output JSON */}
                        <div className="space-y-1.5">
                          <p className="text-[10px] text-slate-500 font-bold uppercase">Final Output / Response</p>
                          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 font-mono text-xs text-slate-300 overflow-x-auto max-h-[160px]">
                            <pre>{JSON.stringify(JSON.parse(selectedSpan.output_json || '{}'), null, 2)}</pre>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="flex-1 flex items-center justify-center text-slate-500 text-xs italic">
                        Select a span from the list to view input/output.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* SCREEN 5: CASE EXPLORER & COMPARISON */}
            {route === 'compare' && compareRunA && compareRunB && (
              <div className="space-y-6">
                {/* Breadcrumbs */}
                <div className="flex items-center space-x-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  <span className="hover:text-white cursor-pointer" onClick={navigateToRuns}>
                    Runs
                  </span>
                  <span>/</span>
                  <span className="text-white">Run Comparison picker</span>
                </div>

                <div className="glass rounded-2xl p-6 relative overflow-hidden">
                  <h2 className="text-xl font-bold text-white flex items-center space-x-2">
                    <TrendingUp className="h-5 w-5 text-indigo-400" />
                    <span>Run A vs Run B Comparison</span>
                  </h2>
                  <p className="text-xs text-slate-400 mt-1 uppercase tracking-wider font-semibold">
                    Compare metric outcomes between baseline and target runs side-by-side
                  </p>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                    <div className="bg-indigo-950/20 border border-indigo-900/30 rounded-xl p-4">
                      <p className="text-[10px] text-indigo-400 font-bold uppercase">Baseline (Run A)</p>
                      <p className="text-sm font-bold text-white mt-1">{compareRunA.target_system_id} ({compareRunA.id.slice(0, 8)})</p>
                      <p className="text-xs text-slate-400 mt-1">Pass Rate: {(compareRunA.overall_pass_rate * 100).toFixed(0)}%</p>
                    </div>
                    <div className="bg-purple-950/20 border border-purple-900/30 rounded-xl p-4">
                      <p className="text-[10px] text-purple-400 font-bold uppercase">Comparison (Run B)</p>
                      <p className="text-sm font-bold text-white mt-1">{compareRunB.target_system_id} ({compareRunB.id.slice(0, 8)})</p>
                      <p className="text-xs text-slate-400 mt-1">Pass Rate: {(compareRunB.overall_pass_rate * 100).toFixed(0)}%</p>
                    </div>
                  </div>
                </div>

                {/* Diffing metric tables */}
                <div className="glass rounded-2xl overflow-hidden">
                  <div className="px-6 py-4 border-b border-slate-800/80 bg-slate-900/40 flex items-center justify-between">
                    <h3 className="text-md font-bold text-white flex items-center space-x-2">
                      <FileText className="h-4 w-4 text-indigo-400" />
                      <span>Metric Delta Comparison</span>
                    </h3>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="border-b border-slate-800/60 bg-slate-950/20 text-slate-400 text-xs font-semibold uppercase">
                          <th className="py-3 px-6">Evaluation Metric</th>
                          <th className="py-3 px-6">User Query Context</th>
                          <th className="py-3 px-6 text-center">Run A Outcome</th>
                          <th className="py-3 px-6 text-center">Run B Outcome</th>
                          <th className="py-3 px-6 text-right">Delta status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/40 text-sm">
                        {compareEvalsA.map((evA) => {
                          const evB = compareEvalsB.find(
                            (b) => b.metric_name === evA.metric_name && b.trace_id === evA.trace_id
                          );
                          const passesA = evA.passed;
                          const passesB = evB ? evB.passed : null;

                          return (
                            <tr key={evA.id} className="hover:bg-slate-900/10">
                              <td className="py-4 px-6 font-semibold text-white">
                                {evA.metric_name}
                              </td>
                              <td className="py-4 px-6 text-xs text-slate-300 max-w-xs truncate" title={evA.root_input || ''}>
                                {evA.root_input || 'N/A'}
                              </td>
                              <td className="py-4 px-6 text-center">
                                <span className={`inline-flex items-center space-x-1 ${passesA ? 'text-emerald-400' : 'text-red-400'}`}>
                                  {passesA ? <CheckCircle2 className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
                                  <span>{passesA ? 'PASS' : 'FAIL'}</span>
                                </span>
                              </td>
                              <td className="py-4 px-6 text-center">
                                {evB ? (
                                  <span className={`inline-flex items-center space-x-1 ${passesB ? 'text-emerald-400' : 'text-red-400'}`}>
                                    {passesB ? <CheckCircle2 className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
                                    <span>{passesB ? 'PASS' : 'FAIL'}</span>
                                  </span>
                                ) : (
                                  <span className="text-slate-500 italic">N/A</span>
                                )}
                              </td>
                              <td className="py-4 px-6 text-right">
                                {passesA && !passesB ? (
                                  <span className="inline-flex items-center space-x-1 text-red-500 font-bold text-xs">
                                    <TrendingDown className="h-4 w-4" />
                                    <span>REGRESSION</span>
                                  </span>
                                ) : !passesA && passesB ? (
                                  <span className="inline-flex items-center space-x-1 text-emerald-500 font-bold text-xs">
                                    <TrendingUp className="h-4 w-4" />
                                    <span>IMPROVEMENT</span>
                                  </span>
                                ) : (
                                  <span className="text-slate-500 text-xs">No Change</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}
