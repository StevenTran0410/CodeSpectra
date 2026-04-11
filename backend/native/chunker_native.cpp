/**
 * chunker_native.cpp — Native merge-pass hotspot for AST chunker (CS-101).
 *
 * merge_spans(spans, target_size) -> list[list[int]]
 *   Greedy accumulation: collect consecutive spans into a group until adding
 *   the next span would push the group's byte-length over target_size.
 *
 * Input:  spans — list of (start_byte: int, end_byte: int, index: int) triples
 *         target_size — max byte length per merged group
 * Output: list of groups; each group is a list of original indices (from the
 *         third element of each input triple), in order.
 */

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

py::list merge_spans(const py::list& spans, int target_size) {
    struct Span {
        int start_byte;
        int end_byte;
        int index;
    };

    std::vector<Span> sv;
    sv.reserve(spans.size());

    for (const auto& item : spans) {
        auto t = py::cast<py::tuple>(item);
        if (t.size() < 3) continue;
        sv.push_back({
            py::cast<int>(t[0]),
            py::cast<int>(t[1]),
            py::cast<int>(t[2]),
        });
    }

    py::list out;
    if (sv.empty()) return out;

    // Greedy merge: accumulate spans into a group while the total byte length
    // (end of last - start of first) stays <= target_size.
    std::vector<int> current_group;
    int group_start = sv[0].start_byte;

    for (const auto& s : sv) {
        int candidate_len = s.end_byte - group_start;
        if (!current_group.empty() && candidate_len > target_size) {
            // Flush current group
            py::list g;
            for (int idx : current_group) g.append(idx);
            out.append(g);
            // Start new group
            current_group.clear();
            group_start = s.start_byte;
        }
        if (current_group.empty()) {
            group_start = s.start_byte;
        }
        current_group.push_back(s.index);
    }

    if (!current_group.empty()) {
        py::list g;
        for (int idx : current_group) g.append(idx);
        out.append(g);
    }

    return out;
}

PYBIND11_MODULE(_native_chunker, m) {
    m.doc() = "Native merge-pass hotspot for AST-based semantic chunker (CS-101)";
    m.def(
        "merge_spans",
        &merge_spans,
        py::arg("spans"),
        py::arg("target_size"),
        "Greedy span merge. spans: list[(start_byte, end_byte, index)]. "
        "Returns list of groups, each group a list of original indices."
    );
}
