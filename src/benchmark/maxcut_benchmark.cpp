#include "sbm/maxcut.hpp"
#include "sbm/solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __linux__
#include <sys/resource.h>
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct MethodResult {
    std::string method;
    double cut = 0.0;
    double milliseconds = 0.0;
    std::string configuration;
};

struct Scale {
    int sbm_steps;
    int sbm_runs;
    int greedy_sweeps;
    int greedy_runs;
    int sa_sweeps;
    int sa_runs;
};

Scale scale_for(std::size_t n) {
    if (n <= 1'024) return {2'000, 8, 50, 8, 200, 8};
    if (n <= 10'000) return {1'000, 4, 30, 4, 100, 4};
    if (n <= 100'000) return {250, 2, 15, 2, 40, 2};
    if (n <= 1'000'000) return {32, 1, 8, 1, 16, 1};
    return {16, 1, 5, 1, 8, 1};
}

double peak_rss_mib() {
#ifdef __linux__
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) return usage.ru_maxrss / 1024.0;
#endif
    return 0.0;
}

double gain_if_flipped(
    std::size_t vertex, const std::vector<std::uint8_t>& partition,
    const sbm::maxcut::Adjacency& adjacency) {
    double gain = 0.0;
    for (std::size_t p = adjacency.row_offsets[vertex];
         p < adjacency.row_offsets[vertex + 1]; ++p) {
        gain += partition[vertex] == partition[adjacency.neighbors[p]]
                    ? adjacency.weights[p]
                    : -adjacency.weights[p];
    }
    return gain;
}

MethodResult exact_maxcut(
    const sbm::maxcut::Graph& graph, const sbm::maxcut::Adjacency& adjacency) {
    const auto start = Clock::now();
    if (graph.vertices > 63) throw std::invalid_argument("exact enumeration supports at most 63 vertices");
    std::vector<std::uint8_t> partition(graph.vertices, 0);
    double current = 0.0;
    double best = 0.0;
    const std::uint64_t states = 1ULL << (graph.vertices - 1);  // fix vertex zero by symmetry.
    for (std::uint64_t state = 1; state < states; ++state) {
        const std::size_t vertex = 1 + static_cast<std::size_t>(__builtin_ctzll(state));
        current += gain_if_flipped(vertex, partition, adjacency);
        partition[vertex] ^= 1;
        best = std::max(best, current);
    }
    const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return {"exact", best, elapsed, "gray-code enumeration"};
}

MethodResult greedy_maxcut(
    const sbm::maxcut::Graph& graph, const sbm::maxcut::Adjacency& adjacency,
    int runs, int max_sweeps, std::uint64_t seed) {
    const auto start = Clock::now();
    std::mt19937_64 rng(seed);
    std::vector<std::uint32_t> order(graph.vertices);
    std::iota(order.begin(), order.end(), 0);
    double best = 0.0;

    for (int run = 0; run < runs; ++run) {
        std::vector<std::uint8_t> partition(graph.vertices);
        for (auto& bit : partition) bit = rng() & 1U;
        double current = graph.cut_value(partition);
        for (int sweep = 0; sweep < max_sweeps; ++sweep) {
            std::shuffle(order.begin(), order.end(), rng);
            bool improved = false;
            for (auto vertex : order) {
                const double gain = gain_if_flipped(vertex, partition, adjacency);
                if (gain > 0.0) {
                    partition[vertex] ^= 1;
                    current += gain;
                    improved = true;
                }
            }
            if (!improved) break;
        }
        best = std::max(best, current);
    }
    const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return {"greedy_local_search", best, elapsed,
            "runs=" + std::to_string(runs) + ";max_sweeps=" + std::to_string(max_sweeps)};
}

MethodResult simulated_annealing(
    const sbm::maxcut::Graph& graph, const sbm::maxcut::Adjacency& adjacency,
    int runs, int sweeps, std::uint64_t seed) {
    const auto start = Clock::now();
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> probability(0.0, 1.0);
    std::uniform_int_distribution<std::uint32_t> vertex(
        0, static_cast<std::uint32_t>(graph.vertices - 1));
    const double total_weight = std::accumulate(
        graph.edges.begin(), graph.edges.end(), 0.0,
        [](double sum, const auto& edge) { return sum + edge.weight; });
    const double initial_temperature = std::max(1.0, 4.0 * total_weight / graph.vertices);
    const double final_temperature = 0.01 * initial_temperature;
    double best = 0.0;

    for (int run = 0; run < runs; ++run) {
        std::vector<std::uint8_t> partition(graph.vertices);
        for (auto& bit : partition) bit = rng() & 1U;
        double current = graph.cut_value(partition);
        best = std::max(best, current);
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            const double fraction = sweeps == 1 ? 1.0 : static_cast<double>(sweep) / (sweeps - 1);
            const double temperature = initial_temperature *
                                       std::pow(final_temperature / initial_temperature, fraction);
            for (std::size_t proposal = 0; proposal < graph.vertices; ++proposal) {
                const auto candidate = vertex(rng);
                const double gain = gain_if_flipped(candidate, partition, adjacency);
                if (gain >= 0.0 || probability(rng) < std::exp(gain / temperature)) {
                    partition[candidate] ^= 1;
                    current += gain;
                    best = std::max(best, current);
                }
            }
        }
    }
    const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return {"simulated_annealing", best, elapsed,
            "runs=" + std::to_string(runs) + ";sweeps=" + std::to_string(sweeps)};
}

MethodResult simulated_bifurcation(
    const sbm::maxcut::Graph& graph, int runs, int steps, std::uint64_t seed) {
    auto bqm = graph.to_bqm();
    sbm::SolverParameters parameters;
    parameters.steps = steps;
    parameters.runs = runs;
    parameters.dt = 1.0;
    parameters.gamma = 0.0;
    parameters.seed = seed;
    const auto start = Clock::now();
    const auto result = sbm::solve_cpu(bqm, parameters);
    const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return {"simulated_bifurcation", graph.cut_value(result.sample), elapsed,
            "runs=" + std::to_string(runs) + ";steps=" + std::to_string(steps)};
}

int main(int argc, char** argv) {
    try {
        const fs::path input_dir = argc > 1 ? argv[1] : "benchmarks/maxcut/instances";
        const fs::path output_path = argc > 2 ? argv[2] : "benchmarks/maxcut/results.csv";
        const std::size_t maximum_vertices = argc > 3
                                                 ? std::stoull(argv[3])
                                                 : std::numeric_limits<std::size_t>::max();
        std::vector<fs::path> instances;
        for (const auto& entry : fs::directory_iterator(input_dir)) {
            if (entry.path().extension() == ".csv" && entry.path().filename() != "manifest.csv") {
                instances.push_back(entry.path());
            }
        }
        std::sort(instances.begin(), instances.end());
        if (!output_path.parent_path().empty()) {
            fs::create_directories(output_path.parent_path());
        }
        std::ofstream output(output_path);
        if (!output) throw std::runtime_error("cannot write benchmark results");
        output << "instance,vertices,edges,load_ms,peak_rss_mib,method,solve_ms,cut,reference_cut,"
                  "quality_ratio,reference_kind,configuration\n";
        output << std::setprecision(17);

        for (const auto& path : instances) {
            const auto load_start = Clock::now();
            auto graph = sbm::maxcut::load_csv(path.string());
            if (graph.vertices > maximum_vertices) continue;
            auto adjacency = sbm::maxcut::make_adjacency(graph);
            const double load_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - load_start).count();
            const auto scale = scale_for(graph.vertices);
            const auto seed = 0x5eedULL + graph.vertices;
            std::cerr << "benchmarking " << path.filename().string() << " (n="
                      << graph.vertices << ", m=" << graph.edges.size() << ")\n";

            std::vector<MethodResult> methods;
            if (graph.vertices <= 24) methods.push_back(exact_maxcut(graph, adjacency));
            methods.push_back(greedy_maxcut(
                graph, adjacency, scale.greedy_runs, scale.greedy_sweeps, seed));
            methods.push_back(simulated_annealing(
                graph, adjacency, scale.sa_runs, scale.sa_sweeps, seed + 1));
            methods.push_back(simulated_bifurcation(
                graph, scale.sbm_runs, scale.sbm_steps, seed + 2));

            const bool has_exact = methods.front().method == "exact";
            const double reference = has_exact
                                         ? methods.front().cut
                                         : std::max_element(
                                               methods.begin(), methods.end(),
                                               [](const auto& a, const auto& b) { return a.cut < b.cut; })
                                               ->cut;
            const double memory_mib = peak_rss_mib();
            for (const auto& method : methods) {
                const double ratio = reference > 0.0 ? method.cut / reference : 1.0;
                output << path.filename().string() << ',' << graph.vertices << ','
                       << graph.edges.size() << ',' << load_ms << ',' << memory_mib << ','
                       << method.method << ','
                       << method.milliseconds << ',' << method.cut << ',' << reference << ','
                       << ratio << ',' << (has_exact ? "exact" : "best_observed") << ','
                       << '"' << method.configuration << '"' << '\n';
                std::cerr << "  " << method.method << ": cut=" << method.cut
                          << ", time_ms=" << method.milliseconds << "\n";
            }
            output.flush();
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
