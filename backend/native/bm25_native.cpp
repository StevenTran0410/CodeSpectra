/**
 * bm25_native.cpp — Native BM25 scorer and impact scorer for CodeSpectra.
 *
 * Module name: _native_bm25
 * Exported functions:
 *   tokenize(text)                     -> list[str]
 *   batch_score(chunks, terms, idf, avgdl, k1, b) -> list[tuple[str, float]]
 *   batch_impact_score(chunks, hop_map, central_ranks, community_map,
 *                      seed_communities, call_chain_files, bm25_scores)
 *                                      -> list[tuple[str, float, float, float, float, float]]
 */

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// ── Tokenizer ──────────────────────────────────────────────────────────────────
// Mirrors Python: re.compile(r"[A-Za-z0-9_]+")
// Splits text into alphanumeric+underscore tokens.
static inline std::vector<std::string> tokenize_str(const std::string& text) {
    std::vector<std::string> tokens;
    const std::size_t n = text.size();
    std::size_t i = 0;
    while (i < n) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (std::isalnum(c) || c == '_') {
            std::size_t j = i + 1;
            while (j < n) {
                const unsigned char d = static_cast<unsigned char>(text[j]);
                if (std::isalnum(d) || d == '_') {
                    ++j;
                } else {
                    break;
                }
            }
            tokens.emplace_back(text.begin() + static_cast<std::ptrdiff_t>(i),
                                text.begin() + static_cast<std::ptrdiff_t>(j));
            i = j;
        } else {
            ++i;
        }
    }
    return tokens;
}

// ── Public: tokenize ───────────────────────────────────────────────────────────
// tokenize(text: str) -> list[str]
py::list tokenize(const std::string& text) {
    auto tokens = tokenize_str(text);
    py::list out;
    out.attr("__init__")();
    for (const auto& tok : tokens) {
        out.append(tok);
    }
    return out;
}

// ── Public: batch_score ────────────────────────────────────────────────────────
// chunks: list of (chunk_id: str, content: str, path: str)
// terms:  list[str]  (pre-lowercased query terms)
// idf:    dict[str, float]
// avgdl:  float
// k1:     float
// b:      float
// Returns: list of (chunk_id: str, score: float)  — all chunks, score may be 0.
py::list batch_score(
    const py::list& chunks,
    const py::list& terms_list,
    const py::dict& idf_dict,
    double avgdl,
    double k1,
    double b
) {
    if (avgdl <= 0.0) avgdl = 1.0;

    // Build term -> IDF map in C++
    std::unordered_map<std::string, double> idf;
    idf.reserve(static_cast<std::size_t>(idf_dict.size()) + 4);
    for (const auto& kv : idf_dict) {
        idf[py::cast<std::string>(kv.first)] = py::cast<double>(kv.second);
    }

    // Collect terms
    std::vector<std::string> terms;
    terms.reserve(static_cast<std::size_t>(terms_list.size()));
    for (const auto& t : terms_list) {
        terms.push_back(py::cast<std::string>(t));
    }

    py::list out;

    for (const auto& item : chunks) {
        auto tup = py::cast<py::tuple>(item);
        if (tup.size() < 3) continue;

        std::string chunk_id = py::cast<std::string>(tup[0]);
        std::string content  = py::cast<std::string>(tup[1]);
        std::string path     = py::cast<std::string>(tup[2]);

        // Lowercase content and path
        std::string content_low = content;
        for (char& ch : content_low) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        std::string path_low = path;
        for (char& ch : path_low) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

        // Tokenize
        auto tokens = tokenize_str(content_low);
        const double dl = static_cast<double>(tokens.size());
        if (dl == 0.0) {
            out.append(py::make_tuple(chunk_id, 0.0));
            continue;
        }

        // Build TF map
        std::unordered_map<std::string, int> tf;
        tf.reserve(tokens.size() + 4);
        for (const auto& tok : tokens) {
            tf[tok] += 1;
        }

        const double length_norm = 1.0 - b + b * dl / avgdl;
        double total = 0.0;

        for (const auto& term : terms) {
            const auto idf_it = idf.find(term);
            if (idf_it == idf.end() || idf_it->second <= 0.0) continue;
            const double idf_val = idf_it->second;

            // Content BM25 contribution
            const auto tf_it = tf.find(term);
            const double f = tf_it != tf.end() ? static_cast<double>(tf_it->second) : 0.0;
            const double content_contrib = idf_val * (f * (k1 + 1.0)) / (f + k1 * length_norm);

            // Path contribution: term present in path -> treat as synthetic doc TF=1
            const double path_contrib = (path_low.find(term) != std::string::npos) ? idf_val : 0.0;

            total += content_contrib + path_contrib;
        }

        out.append(py::make_tuple(chunk_id, total));
    }

    return out;
}

// ── Scoring constants (mirrors impact_retrieval.py) ───────────────────────────

static const double HOP_WEIGHT_0 = 3.0;
static const double HOP_WEIGHT_1 = 2.0;
static const double HOP_WEIGHT_2 = 1.2;
static const double HOP_WEIGHT_3 = 0.6;
static const double HOP_WEIGHT_4 = 0.3;
static const double HOP_WEIGHT_INF = 0.2;  // hop > 4

static const double IMPACT_CENTRALITY_BONUS = 2.0;
static const double IMPACT_CENTRALITY_DECAY = 0.06;
static const double IMPACT_COMMUNITY_BONUS = 1.0;
static const double IMPACT_NEIGHBOR_COMMUNITY_BONUS = 0.4;
static const double IMPACT_SYMBOL_CHAIN_BONUS = 2.5;
static const double IMPACT_BM25_WEIGHT = 0.3;

static inline double hop_weight(int hop) {
    switch (hop) {
        case 0: return HOP_WEIGHT_0;
        case 1: return HOP_WEIGHT_1;
        case 2: return HOP_WEIGHT_2;
        case 3: return HOP_WEIGHT_3;
        case 4: return HOP_WEIGHT_4;
        default: return HOP_WEIGHT_INF;
    }
}

// ── Public: batch_impact_score ─────────────────────────────────────────────────
// chunks:            list of dicts with keys: id, rel_path
// hop_map:           dict[str, int]     rel_path -> hop distance
// central_ranks:     dict[str, int]     rel_path -> rank index
// community_map:     dict[str, int]     rel_path -> community_id
// seed_communities:  set[int]           community IDs of seed files
// call_chain_files:  set[str]           rel_paths appearing in call chain
// bm25_scores:       dict[str, float]   chunk_id -> bm25 score
//
// Returns: list of (chunk_id, total, bm25_boost, symbol_bonus, community_bonus, centrality_bonus)
py::list batch_impact_score(
    const py::list& chunks,
    const py::dict& hop_map_dict,
    const py::dict& central_ranks_dict,
    const py::dict& community_map_dict,
    const py::set& seed_communities_set,
    const py::set& call_chain_files_set,
    const py::dict& bm25_scores_dict
) {
    // Build C++ structures for O(1) lookups
    std::unordered_map<std::string, int> hop_map;
    hop_map.reserve(static_cast<std::size_t>(hop_map_dict.size()) + 4);
    for (const auto& kv : hop_map_dict) {
        hop_map[py::cast<std::string>(kv.first)] = py::cast<int>(kv.second);
    }

    std::unordered_map<std::string, int> central_ranks;
    central_ranks.reserve(static_cast<std::size_t>(central_ranks_dict.size()) + 4);
    for (const auto& kv : central_ranks_dict) {
        central_ranks[py::cast<std::string>(kv.first)] = py::cast<int>(kv.second);
    }

    std::unordered_map<std::string, int> community_map;
    community_map.reserve(static_cast<std::size_t>(community_map_dict.size()) + 4);
    for (const auto& kv : community_map_dict) {
        community_map[py::cast<std::string>(kv.first)] = py::cast<int>(kv.second);
    }

    std::unordered_set<int> seed_communities;
    seed_communities.reserve(static_cast<std::size_t>(seed_communities_set.size()) + 4);
    for (const auto& item : seed_communities_set) {
        seed_communities.insert(py::cast<int>(item));
    }

    std::unordered_set<std::string> call_chain_files;
    call_chain_files.reserve(static_cast<std::size_t>(call_chain_files_set.size()) + 4);
    for (const auto& item : call_chain_files_set) {
        call_chain_files.insert(py::cast<std::string>(item));
    }

    std::unordered_map<std::string, double> bm25_scores;
    bm25_scores.reserve(static_cast<std::size_t>(bm25_scores_dict.size()) + 4);
    for (const auto& kv : bm25_scores_dict) {
        bm25_scores[py::cast<std::string>(kv.first)] = py::cast<double>(kv.second);
    }

    py::list out;

    for (const auto& chunk_obj : chunks) {
        auto chunk = py::cast<py::dict>(chunk_obj);
        std::string chunk_id = py::cast<std::string>(chunk["id"]);
        std::string rel_path = py::cast<std::string>(chunk["rel_path"]);

        // Hop weight
        double hw = 0.0;
        {
            const auto it = hop_map.find(rel_path);
            if (it != hop_map.end()) {
                hw = hop_weight(it->second);
            }
        }

        // BM25 boost
        double bm25_score = 0.0;
        {
            const auto it = bm25_scores.find(chunk_id);
            if (it != bm25_scores.end()) bm25_score = it->second;
        }
        const double bm25_boost = bm25_score * IMPACT_BM25_WEIGHT;

        // Symbol chain bonus
        const double symbol_bonus = call_chain_files.count(rel_path) ? IMPACT_SYMBOL_CHAIN_BONUS : 0.0;

        // Community bonus
        double community_bonus = 0.0;
        {
            const auto cit = community_map.find(rel_path);
            if (cit != community_map.end()) {
                if (seed_communities.count(cit->second)) {
                    community_bonus = IMPACT_COMMUNITY_BONUS;
                }
                // neighbor community bonus is not applied here (requires neighbor_community_ids
                // which is a separate set not passed to this function — caller handles it)
            }
        }

        // Centrality bonus with exponential decay
        double centrality_bonus = 0.0;
        {
            const auto rit = central_ranks.find(rel_path);
            if (rit != central_ranks.end()) {
                centrality_bonus = IMPACT_CENTRALITY_BONUS *
                    std::pow(1.0 - IMPACT_CENTRALITY_DECAY, static_cast<double>(rit->second));
            }
        }

        const double total = hw + bm25_boost + symbol_bonus + community_bonus + centrality_bonus;

        out.append(py::make_tuple(chunk_id, total, bm25_boost, symbol_bonus, community_bonus, centrality_bonus));
    }

    return out;
}

PYBIND11_MODULE(_native_bm25, m) {
    m.doc() = "Native BM25 scorer and impact scorer for CodeSpectra";
    m.def(
        "tokenize",
        &tokenize,
        py::arg("text"),
        "Tokenize text using [A-Za-z0-9_]+ regex (mirrors Python BM25Scorer._WORD)"
    );
    m.def(
        "batch_score",
        &batch_score,
        py::arg("chunks"),
        py::arg("terms"),
        py::arg("idf"),
        py::arg("avgdl"),
        py::arg("k1"),
        py::arg("b"),
        "Score multiple chunks in one native call. "
        "chunks: list[(chunk_id, content, path)]. "
        "Returns list[(chunk_id, score)]."
    );
    m.def(
        "batch_impact_score",
        &batch_impact_score,
        py::arg("chunks"),
        py::arg("hop_map"),
        py::arg("central_ranks"),
        py::arg("community_map"),
        py::arg("seed_communities"),
        py::arg("call_chain_files"),
        py::arg("bm25_scores"),
        "Score chunks for impact retrieval in one native call. "
        "Returns list[(chunk_id, total, bm25_boost, symbol_bonus, community_bonus, centrality_bonus)]."
    );
}
