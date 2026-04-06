#include <algorithm>
#include <numeric>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

struct Edge {
    std::string src;
    std::string dst;
    std::string type;
    bool is_external;
};

std::string other_node(const Edge& e, const std::string& node) {
    if (e.src == node) {
        return e.dst;
    }
    return e.src;
}

}  // namespace

py::list compute_scores(const std::vector<std::string>& nodes, const py::list& edges) {
    std::unordered_map<std::string, int> indeg;
    std::unordered_map<std::string, int> outdeg;
    indeg.reserve(nodes.size() * 2 + 32);
    outdeg.reserve(nodes.size() * 2 + 32);

    for (const auto& n : nodes) {
        indeg[n] = 0;
        outdeg[n] = 0;
    }

    for (const auto& item : edges) {
        auto t = py::cast<py::tuple>(item);
        if (t.size() < 2) {
            continue;
        }
        auto src = py::cast<std::string>(t[0]);
        auto dst = py::cast<std::string>(t[1]);
        outdeg[src] += 1;
        indeg[dst] += 1;
    }

    struct ScoreItem {
        std::string rel_path;
        int indegree;
        int outdegree;
        int score;
    };

    std::vector<ScoreItem> items;
    items.reserve(indeg.size() + 16);

    std::unordered_set<std::string> seen;
    seen.reserve(indeg.size() + outdeg.size() + 16);

    for (const auto& kv : indeg) {
        seen.insert(kv.first);
    }
    for (const auto& kv : outdeg) {
        seen.insert(kv.first);
    }

    for (const auto& node : seen) {
        const int in_v = indeg.count(node) ? indeg[node] : 0;
        const int out_v = outdeg.count(node) ? outdeg[node] : 0;
        items.push_back(ScoreItem{node, in_v, out_v, in_v * 3 + out_v});
    }

    std::sort(items.begin(), items.end(), [](const ScoreItem& a, const ScoreItem& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.indegree != b.indegree) return a.indegree > b.indegree;
        return a.rel_path < b.rel_path;
    });

    py::list out;
    for (const auto& it : items) {
        py::dict d;
        d["rel_path"] = it.rel_path;
        d["indegree"] = it.indegree;
        d["outdegree"] = it.outdegree;
        d["score"] = it.score;
        out.append(d);
    }
    return out;
}

py::dict expand_neighbors(
    const std::string& seed,
    const py::list& edge_tuples,
    int hops,
    int limit
) {
    if (hops < 1) hops = 1;
    if (hops > 4) hops = 4;
    if (limit < 10) limit = 10;
    if (limit > 2000) limit = 2000;

    std::vector<Edge> edges;
    edges.reserve(edge_tuples.size());

    for (const auto& item : edge_tuples) {
        auto t = py::cast<py::tuple>(item);
        if (t.size() < 4) {
            continue;
        }
        Edge e;
        e.src = py::cast<std::string>(t[0]);
        e.dst = py::cast<std::string>(t[1]);
        e.type = py::cast<std::string>(t[2]);
        e.is_external = py::cast<int>(t[3]) != 0;
        if (!e.is_external) {
            edges.push_back(std::move(e));
        }
    }

    std::unordered_map<std::string, std::vector<int>> adjacency;
    adjacency.reserve(edges.size() * 2 + 16);
    for (int i = 0; i < static_cast<int>(edges.size()); i++) {
        adjacency[edges[i].src].push_back(i);
        adjacency[edges[i].dst].push_back(i);
    }

    std::unordered_set<std::string> visited;
    visited.reserve(limit + 16);
    visited.insert(seed);

    std::unordered_set<int> kept_edge_indexes;
    kept_edge_indexes.reserve(limit + 16);

    std::unordered_set<std::string> frontier;
    frontier.insert(seed);

    for (int h = 0; h < hops; h++) {
        std::unordered_set<std::string> next_frontier;
        for (const auto& node : frontier) {
            const auto it = adjacency.find(node);
            if (it == adjacency.end()) {
                continue;
            }
            for (const int ei : it->second) {
                if (static_cast<int>(kept_edge_indexes.size()) < limit) {
                    kept_edge_indexes.insert(ei);
                }
                const auto nxt = other_node(edges[ei], node);
                if (visited.count(nxt) == 0 && static_cast<int>(visited.size()) < limit) {
                    visited.insert(nxt);
                    next_frontier.insert(nxt);
                }
            }
        }
        if (next_frontier.empty()) {
            break;
        }
        frontier = std::move(next_frontier);
    }

    std::vector<std::string> nodes_vec;
    nodes_vec.reserve(visited.size());
    for (const auto& n : visited) {
        nodes_vec.push_back(n);
    }
    std::sort(nodes_vec.begin(), nodes_vec.end());

    py::list out_edges;
    for (const int ei : kept_edge_indexes) {
        py::tuple t(3);
        t[0] = edges[ei].src;
        t[1] = edges[ei].dst;
        t[2] = edges[ei].type;
        out_edges.append(t);
    }

    py::dict out;
    out["nodes"] = nodes_vec;
    out["edges"] = out_edges;
    return out;
}

// ── Tarjan SCC ───────────────────────────────────────────────────────────────
// Returns all strongly-connected components with ≥ 2 nodes (circular import candidates).
// Input:  edge_tuples — list of (src:str, dst:str) tuples (internal edges only)
// Output: list of lists of str (each SCC sorted, result sorted by size desc)
py::list compute_scc(const py::list& edge_tuples) {
    // Build string → int ID mapping
    std::unordered_map<std::string, int> id_of;
    std::vector<std::string>             id_name;
    id_of.reserve(edge_tuples.size() * 2 + 64);

    auto intern = [&](const std::string& s) -> int {
        auto it = id_of.find(s);
        if (it != id_of.end()) return it->second;
        int id = static_cast<int>(id_name.size());
        id_of[s] = id;
        id_name.push_back(s);
        return id;
    };

    // Collect directed edges (src → dst)
    struct DirEdge { int src, dst; };
    std::vector<DirEdge> edges;
    edges.reserve(edge_tuples.size());
    for (const auto& item : edge_tuples) {
        auto t = py::cast<py::tuple>(item);
        if (t.size() < 2) continue;
        int s = intern(py::cast<std::string>(t[0]));
        int d = intern(py::cast<std::string>(t[1]));
        if (s != d) edges.push_back({s, d});
    }

    int N = static_cast<int>(id_name.size());
    if (N == 0) return py::list{};

    // Adjacency list
    std::vector<std::vector<int>> adj(N);
    for (const auto& e : edges) adj[e.src].push_back(e.dst);

    // Iterative Tarjan
    std::vector<int> index(N, -1), lowlink(N, 0), comp(N, -1);
    std::vector<bool> on_stack(N, false);
    std::vector<int> stk;
    stk.reserve(N);
    int timer = 0;
    std::vector<std::vector<int>> sccs;

    // explicit call stack to avoid C++ stack overflow on deep graphs
    struct Frame { int v, ei; };
    std::vector<Frame> call_stack;
    call_stack.reserve(N);

    for (int root = 0; root < N; ++root) {
        if (index[root] != -1) continue;
        call_stack.push_back({root, 0});
        index[root] = lowlink[root] = timer++;
        stk.push_back(root);
        on_stack[root] = true;

        while (!call_stack.empty()) {
            auto& [v, ei] = call_stack.back();
            if (ei < static_cast<int>(adj[v].size())) {
                int w = adj[v][ei++];
                if (index[w] == -1) {
                    index[w] = lowlink[w] = timer++;
                    stk.push_back(w);
                    on_stack[w] = true;
                    call_stack.push_back({w, 0});
                } else if (on_stack[w]) {
                    lowlink[v] = std::min(lowlink[v], index[w]);
                }
            } else {
                // Done with v
                call_stack.pop_back();
                if (!call_stack.empty()) {
                    int parent = call_stack.back().v;
                    lowlink[parent] = std::min(lowlink[parent], lowlink[v]);
                }
                if (lowlink[v] == index[v]) {
                    std::vector<int> scc;
                    while (true) {
                        int w = stk.back(); stk.pop_back();
                        on_stack[w] = false;
                        comp[w] = static_cast<int>(sccs.size());
                        scc.push_back(w);
                        if (w == v) break;
                    }
                    if (scc.size() >= 2) sccs.push_back(std::move(scc));
                }
            }
        }
    }

    // Sort each SCC by name, then sort SCCs by size (largest first)
    std::sort(sccs.begin(), sccs.end(), [](const auto& a, const auto& b) {
        return a.size() > b.size();
    });

    py::list out;
    for (const auto& scc : sccs) {
        std::vector<std::string> names;
        names.reserve(scc.size());
        for (int id : scc) names.push_back(id_name[id]);
        std::sort(names.begin(), names.end());
        out.append(names);
    }
    return out;
}

// ── Bulk keyword scanner ──────────────────────────────────────────────────────
// For each (rel_path, content) pair, counts occurrences of any keyword in the list.
// Returns a list of {rel_path, count} dicts — only entries where count > 0.
// keywords: list of plain strings (case-sensitive exact substring match).
// Use uppercase keywords; caller is responsible for case normalisation if needed.
py::list scan_keywords_bulk(const py::list& chunks, const py::list& keywords) {
    // Pre-build keyword list
    std::vector<std::string> kws;
    kws.reserve(keywords.size());
    for (const auto& k : keywords) kws.push_back(py::cast<std::string>(k));

    py::list out;
    for (const auto& chunk : chunks) {
        auto t = py::cast<py::tuple>(chunk);
        if (t.size() < 2) continue;
        std::string rel_path = py::cast<std::string>(t[0]);
        std::string content  = py::cast<std::string>(t[1]);

        int total = 0;
        for (const auto& kw : kws) {
            std::size_t pos = 0;
            while ((pos = content.find(kw, pos)) != std::string::npos) {
                // Word-boundary check: preceding and following char must not be alnum/_
                bool pre_ok = (pos == 0) || (!std::isalnum(static_cast<unsigned char>(content[pos - 1]))
                                             && content[pos - 1] != '_');
                bool suf_ok = (pos + kw.size() >= content.size())
                              || (!std::isalnum(static_cast<unsigned char>(content[pos + kw.size()]))
                                  && content[pos + kw.size()] != '_');
                if (pre_ok && suf_ok) ++total;
                pos += kw.size();
            }
        }
        if (total > 0) {
            py::dict d;
            d["rel_path"] = rel_path;
            d["count"]    = total;
            out.append(d);
        }
    }
    return out;
}

PYBIND11_MODULE(_native_graph, m) {
    m.doc() = "Native graph hotspot module for CodeSpectra";
    m.def("compute_scores",      &compute_scores,      "Compute graph centrality score list");
    m.def("expand_neighbors",    &expand_neighbors,    "Expand graph neighborhood");
    m.def("compute_scc",         &compute_scc,         "Tarjan SCC — returns circular import cycles (size >= 2)");
    m.def("scan_keywords_bulk",  &scan_keywords_bulk,  "Bulk keyword scanner — word-boundary aware, returns [{rel_path, count}]");
}
