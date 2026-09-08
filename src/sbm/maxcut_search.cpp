#include "sbm/maxcut.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace sbm::maxcut {
PreparedSearch::PreparedSearch(const Graph& graph) : graph_(graph) {
    if (graph.vertices == 0) throw std::invalid_argument("MaxCut graph must not be empty");
    for (const auto& edge : graph.edges)
        if (edge.u >= graph.vertices || edge.v >= graph.vertices || edge.u == edge.v || !std::isfinite(edge.weight))
            throw std::invalid_argument("invalid MaxCut edge");
    adjacency_ = make_adjacency(graph);
}

double PreparedSearch::gain(std::size_t vertex, const std::vector<std::uint8_t>& partition) const {
    double value = 0.0;
    for (auto p = adjacency_.row_offsets[vertex]; p < adjacency_.row_offsets[vertex + 1]; ++p)
        value += partition[vertex] == partition[adjacency_.neighbors[p]] ? adjacency_.weights[p] : -adjacency_.weights[p];
    return value;
}

SearchResult PreparedSearch::exact() const {
    if (graph_.vertices > 63) throw std::invalid_argument("exact enumeration supports at most 63 vertices");
    std::vector<std::uint8_t> partition(graph_.vertices, 0);
    SearchResult best{partition, 0.0};
    double current = 0.0;
    const std::uint64_t states = 1ULL << (graph_.vertices - 1);
    for (std::uint64_t state = 1; state < states; ++state) {
        const auto vertex = 1 + static_cast<std::size_t>(__builtin_ctzll(state));
        current += gain(vertex, partition);
        partition[vertex] ^= 1;
        if (current > best.cut) best = {partition, current};
    }
    best.cut = graph_.cut_value(best.partition);
    return best;
}

SearchResult PreparedSearch::greedy(int runs, int sweeps, std::uint64_t seed) const {
    if (runs <= 0 || sweeps <= 0) throw std::invalid_argument("MaxCut runs and sweeps must be positive");
    std::mt19937_64 rng(seed);
    std::vector<std::uint32_t> order(graph_.vertices);
    std::iota(order.begin(), order.end(), 0);
    SearchResult best{std::vector<std::uint8_t>(graph_.vertices, 0), 0.0};
    for (int run = 0; run < runs; ++run) {
        std::vector<std::uint8_t> partition(graph_.vertices);
        for (auto& bit : partition) bit = rng() & 1U;
        double current = graph_.cut_value(partition);
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            std::shuffle(order.begin(), order.end(), rng);
            bool changed = false;
            for (auto vertex : order) {
                const auto delta = gain(vertex, partition);
                if (delta > 0.0) { partition[vertex] ^= 1; current += delta; changed = true; }
            }
            if (!changed) break;
        }
        if (current > best.cut) best = {partition, current};
    }
    best.cut = graph_.cut_value(best.partition);
    return best;
}

SearchResult PreparedSearch::anneal(int runs, int sweeps, std::uint64_t seed) const {
    if (runs <= 0 || sweeps <= 0) throw std::invalid_argument("MaxCut runs and sweeps must be positive");
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> probability(0.0, 1.0);
    std::uniform_int_distribution<std::uint32_t> vertex(0, static_cast<std::uint32_t>(graph_.vertices - 1));
    const double total_weight = std::accumulate(graph_.edges.begin(), graph_.edges.end(), 0.0,
        [](double sum, const auto& edge) { return sum + edge.weight; });
    const double t0 = std::max(1.0, 4.0 * total_weight / graph_.vertices);
    const double t1 = 0.01 * t0;
    SearchResult best{std::vector<std::uint8_t>(graph_.vertices, 0), 0.0};
    for (int run = 0; run < runs; ++run) {
        std::vector<std::uint8_t> partition(graph_.vertices);
        for (auto& bit : partition) bit = rng() & 1U;
        double current = graph_.cut_value(partition);
        std::vector<std::uint32_t> pending;
        auto compact = [&] {
            pending.clear();
            for (std::size_t i = 0; i < partition.size(); ++i)
                if (partition[i] != best.partition[i]) pending.push_back(static_cast<std::uint32_t>(i));
        };
        compact();
        auto record = [&] {
            if (current > best.cut) {
                for (auto i : pending) best.partition[i] ^= 1;
                best.cut = current;
                pending.clear();
            } else if (pending.size() > graph_.vertices * 2) {
                compact();
            }
        };
        record();
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            const double fraction = sweeps == 1 ? 1.0 : static_cast<double>(sweep) / (sweeps - 1);
            const double temperature = t0 * std::pow(t1 / t0, fraction);
            for (std::size_t proposal = 0; proposal < graph_.vertices; ++proposal) {
                const auto candidate = vertex(rng);
                const auto delta = gain(candidate, partition);
                if (delta >= 0.0 || probability(rng) < std::exp(delta / temperature)) {
                    partition[candidate] ^= 1; current += delta; pending.push_back(candidate); record();
                }
            }
        }
    }
    best.cut = graph_.cut_value(best.partition);
    return best;
}
} // namespace sbm::maxcut
