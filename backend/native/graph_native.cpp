#include <algorithm>
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

PYBIND11_MODULE(_native_graph, m) {
    m.doc() = "Native graph hotspot module for CodeSpectra";
    m.def("compute_scores", &compute_scores, "Compute graph centrality score list");
    m.def("expand_neighbors", &expand_neighbors, "Expand graph neighborhood");
}
