#include "sbm/maxcut.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace sbm::maxcut {

double Graph::cut_value(const std::vector<std::uint8_t>& partition) const {
    if (partition.size() != vertices) {
        throw std::invalid_argument("partition size does not match graph");
    }
    double cut = 0.0;
    for (const auto& edge : edges) {
        if (partition[edge.u] != partition[edge.v]) cut += edge.weight;
    }
    return cut;
}

BinaryQuadraticModel Graph::to_bqm() const {
    BinaryQuadraticModel bqm;
    bqm.linear.assign(vertices, 0.0);
    bqm.quadratic.reserve(edges.size());
    for (const auto& edge : edges) {
        bqm.linear[edge.u] -= edge.weight;
        bqm.linear[edge.v] -= edge.weight;
        bqm.quadratic.push_back({edge.u, edge.v, 2.0 * edge.weight});
    }
    return bqm;
}

Graph generate(
    std::size_t vertices, std::size_t edge_count, std::uint64_t seed,
    int minimum_weight, int maximum_weight) {
    if (vertices < 2 || vertices > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("vertices must fit in uint32 and be at least two");
    }
    if (minimum_weight <= 0 || minimum_weight > maximum_weight) {
        throw std::invalid_argument("invalid positive weight range");
    }
    const auto max_edges = vertices * (vertices - 1) / 2;
    if (edge_count > max_edges) edge_count = max_edges;

    Graph graph;
    graph.vertices = vertices;
    graph.edges.reserve(edge_count);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint32_t> vertex(
        0, static_cast<std::uint32_t>(vertices - 1));
    std::uniform_int_distribution<int> weight(minimum_weight, maximum_weight);

    // For RAM-scale cases, the duplicate probability is negligible. Parallel
    // edges are valid weighted MaxCut terms and avoid a large hash table.
    while (graph.edges.size() < edge_count) {
        auto u = vertex(rng);
        auto v = vertex(rng);
        if (u == v) continue;
        if (u > v) std::swap(u, v);
        graph.edges.push_back({u, v, static_cast<double>(weight(rng))});
    }
    return graph;
}

Graph load_csv(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open MaxCut CSV: " + path);
    Graph graph;
    std::string line;
    std::uint32_t largest_vertex = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        if (line.rfind("# vertices=", 0) == 0) {
            graph.vertices = std::stoull(line.substr(11));
            continue;
        }
        if (line[0] == '#' || line.rfind("u,v,weight", 0) == 0) continue;

        char* end = nullptr;
        const char* cursor = line.c_str();
        const auto u = static_cast<std::uint32_t>(std::strtoul(cursor, &end, 10));
        if (*end != ',') throw std::runtime_error("malformed MaxCut CSV row");
        cursor = end + 1;
        const auto v = static_cast<std::uint32_t>(std::strtoul(cursor, &end, 10));
        if (*end != ',') throw std::runtime_error("malformed MaxCut CSV row");
        cursor = end + 1;
        const double weight = std::strtod(cursor, &end);
        if (end == cursor || weight <= 0.0 || u == v) {
            throw std::runtime_error("invalid MaxCut edge");
        }
        graph.edges.push_back({u, v, weight});
        largest_vertex = std::max(largest_vertex, std::max(u, v));
    }
    if (graph.vertices == 0) graph.vertices = static_cast<std::size_t>(largest_vertex) + 1;
    if (largest_vertex >= graph.vertices) {
        throw std::runtime_error("edge endpoint exceeds declared vertex count");
    }
    return graph;
}

void save_csv(const Graph& graph, const std::string& path, std::uint64_t seed) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write MaxCut CSV: " + path);
    static char buffer[1 << 20];
    output.rdbuf()->pubsetbuf(buffer, sizeof(buffer));
    output << "# maxcut_csv_v1\n# vertices=" << graph.vertices << "\n# seed=" << seed
           << "\nu,v,weight\n";
    output << std::setprecision(17);
    for (const auto& edge : graph.edges) {
        output << edge.u << ',' << edge.v << ',' << edge.weight << '\n';
    }
}

Adjacency make_adjacency(const Graph& graph) {
    Adjacency adjacency;
    adjacency.row_offsets.assign(graph.vertices + 1, 0);
    for (const auto& edge : graph.edges) {
        ++adjacency.row_offsets[edge.u + 1];
        ++adjacency.row_offsets[edge.v + 1];
    }
    for (std::size_t i = 0; i < graph.vertices; ++i) {
        adjacency.row_offsets[i + 1] += adjacency.row_offsets[i];
    }
    adjacency.neighbors.resize(adjacency.row_offsets.back());
    adjacency.weights.resize(adjacency.row_offsets.back());
    auto cursor = adjacency.row_offsets;
    for (const auto& edge : graph.edges) {
        auto uv = cursor[edge.u]++;
        auto vu = cursor[edge.v]++;
        adjacency.neighbors[uv] = edge.v;
        adjacency.weights[uv] = edge.weight;
        adjacency.neighbors[vu] = edge.u;
        adjacency.weights[vu] = edge.weight;
    }
    return adjacency;
}

}  // namespace sbm::maxcut
