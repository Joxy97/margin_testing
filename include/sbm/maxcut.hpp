#pragma once

#include "sbm/model.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sbm::maxcut {

struct Edge {
    std::uint32_t u;
    std::uint32_t v;
    double weight;
};

struct Graph {
    std::size_t vertices = 0;
    std::vector<Edge> edges;

    [[nodiscard]] double cut_value(const std::vector<std::uint8_t>& partition) const;
    [[nodiscard]] BinaryQuadraticModel to_bqm() const;
};

struct Adjacency {
    std::vector<std::size_t> row_offsets;
    std::vector<std::uint32_t> neighbors;
    std::vector<double> weights;
};

[[nodiscard]] Graph generate(
    std::size_t vertices, std::size_t edges, std::uint64_t seed,
    int minimum_weight = 1, int maximum_weight = 10);
[[nodiscard]] Graph load_csv(const std::string& path);
void save_csv(const Graph& graph, const std::string& path, std::uint64_t seed);
[[nodiscard]] Adjacency make_adjacency(const Graph& graph);

}  // namespace sbm::maxcut
