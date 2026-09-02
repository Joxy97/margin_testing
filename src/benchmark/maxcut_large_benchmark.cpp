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
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __linux__
#include <sys/resource.h>
#endif

using Clock = std::chrono::steady_clock;

struct Result {
    double cut;
    double milliseconds;
};

void enforce_four_gib_limit() {
#ifdef __linux__
    constexpr rlim_t four_gib = 4ULL * 1024 * 1024 * 1024;
    const rlimit limit{four_gib, four_gib};
    if (setrlimit(RLIMIT_AS, &limit) != 0) {
        throw std::runtime_error("failed to apply the 4 GiB address-space limit");
    }
#else
    throw std::runtime_error("the hard memory limit is implemented for Linux");
#endif
}

double peak_rss_mib() {
#ifdef __linux__
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) return usage.ru_maxrss / 1024.0;
#endif
    return 0.0;
}

double gain(
    std::size_t vertex, const std::vector<std::uint8_t>& partition,
    const sbm::maxcut::Adjacency& adjacency) {
    double value = 0.0;
    for (std::size_t p = adjacency.row_offsets[vertex];
         p < adjacency.row_offsets[vertex + 1]; ++p) {
        value += partition[vertex] == partition[adjacency.neighbors[p]]
                     ? adjacency.weights[p]
                     : -adjacency.weights[p];
    }
    return value;
}

Result greedy(const sbm::maxcut::Graph& graph, int sweeps, std::uint64_t seed) {
    const auto start = Clock::now();
    auto adjacency = sbm::maxcut::make_adjacency(graph);
    std::mt19937_64 rng(seed);
    std::vector<std::uint8_t> partition(graph.vertices);
    for (auto& bit : partition) bit = rng() & 1U;
    std::vector<std::uint32_t> order(graph.vertices);
    std::iota(order.begin(), order.end(), 0);
    double cut = graph.cut_value(partition);
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        std::shuffle(order.begin(), order.end(), rng);
        bool changed = false;
        for (auto vertex : order) {
            const double delta = gain(vertex, partition, adjacency);
            if (delta > 0.0) {
                partition[vertex] ^= 1;
                cut += delta;
                changed = true;
            }
        }
        if (!changed) break;
    }
    return {cut, std::chrono::duration<double, std::milli>(Clock::now() - start).count()};
}

Result anneal(const sbm::maxcut::Graph& graph, int sweeps, std::uint64_t seed) {
    const auto start = Clock::now();
    auto adjacency = sbm::maxcut::make_adjacency(graph);
    std::mt19937_64 rng(seed);
    std::vector<std::uint8_t> partition(graph.vertices);
    for (auto& bit : partition) bit = rng() & 1U;
    std::uniform_real_distribution<double> probability(0.0, 1.0);
    std::uniform_int_distribution<std::uint32_t> vertex(
        0, static_cast<std::uint32_t>(graph.vertices - 1));
    const double total_weight = std::accumulate(
        graph.edges.begin(), graph.edges.end(), 0.0,
        [](double sum, const auto& edge) { return sum + edge.weight; });
    const double t0 = std::max(1.0, 4.0 * total_weight / graph.vertices);
    const double t1 = 0.01 * t0;
    double cut = graph.cut_value(partition);
    double best = cut;
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        const double fraction = sweeps == 1 ? 1.0 : static_cast<double>(sweep) / (sweeps - 1);
        const double temperature = t0 * std::pow(t1 / t0, fraction);
        for (std::size_t proposal = 0; proposal < graph.vertices; ++proposal) {
            const auto candidate = vertex(rng);
            const double delta = gain(candidate, partition, adjacency);
            if (delta >= 0.0 || probability(rng) < std::exp(delta / temperature)) {
                partition[candidate] ^= 1;
                cut += delta;
                best = std::max(best, cut);
            }
        }
    }
    return {best, std::chrono::duration<double, std::milli>(Clock::now() - start).count()};
}

int main(int argc, char** argv) {
    try {
        enforce_four_gib_limit();
        const std::filesystem::path output_path =
            argc > 1 ? argv[1] : "benchmarks/maxcut_large/results.csv";
        const int qubos_per_size = argc > 2 ? std::stoi(argv[2]) : 10;
        if (qubos_per_size != 10) {
            throw std::invalid_argument("the large benchmark requires exactly 10 QUBOs per size");
        }
        if (!output_path.parent_path().empty()) {
            std::filesystem::create_directories(output_path.parent_path());
        }
        std::ofstream output(output_path);
        if (!output) throw std::runtime_error("cannot write large benchmark results");
        output << "vertices,edges,qubo,seed,method,solve_ms,cut,reference_cut,quality_ratio,"
                  "peak_rss_mib,configuration\n";
        output << std::setprecision(17);

        const std::vector<std::size_t> sizes{3'000'000, 5'000'000, 10'000'000};
        constexpr std::uint64_t base_seed = 0x4'4742'2026ULL;
        for (auto vertices : sizes) {
            const std::size_t edges = vertices * 2;  // average degree four.
            constexpr int dsb_steps = 30;
            constexpr int greedy_sweeps = 5;
            constexpr int sa_sweeps = 8;
            std::cerr << "large batch: n=" << vertices << ", count=10\n";

            for (int qubo = 0; qubo < qubos_per_size; ++qubo) {
                const auto seed = base_seed + vertices + qubo;
                std::cerr << "  QUBO " << (qubo + 1) << "/10" << std::flush;
                auto graph = sbm::maxcut::generate(vertices, edges, seed);
                const auto greedy_result = greedy(graph, greedy_sweeps, seed + 1);
                const auto sa_result = anneal(graph, sa_sweeps, seed + 2);

                auto bqm = graph.to_bqm();
                graph = {};
                graph.edges.shrink_to_fit();
                sbm::SolverParameters parameters;
                parameters.steps = dsb_steps;
                parameters.runs = 1;
                parameters.seed = seed + 3;
                const auto start = Clock::now();
                const auto dsb_result = sbm::solve_cpu(bqm, parameters);
                const double dsb_ms =
                    std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                const double dsb_cut = -dsb_result.energy;
                const double reference = std::max({greedy_result.cut, sa_result.cut, dsb_cut});
                const double memory = peak_rss_mib();

                const auto write = [&](const char* method, const Result& result,
                                       const std::string& configuration) {
                    output << vertices << ',' << edges << ',' << qubo << ',' << seed << ','
                           << method << ',' << result.milliseconds << ',' << result.cut << ','
                           << reference << ',' << result.cut / reference << ',' << memory << ','
                           << '"' << configuration << '"' << '\n';
                };
                write("greedy_local_search", greedy_result,
                      "runs=1;max_sweeps=" + std::to_string(greedy_sweeps));
                write("simulated_annealing", sa_result,
                      "runs=1;sweeps=" + std::to_string(sa_sweeps));
                write("simulated_bifurcation", {dsb_cut, dsb_ms},
                      "runs=1;steps=" + std::to_string(dsb_steps) +
                          ";openmp_rows=on;simd=on");
                output.flush();
                std::cerr << ": dSB=" << dsb_ms << " ms, RSS=" << memory << " MiB\n";
            }
        }
    } catch (const std::bad_alloc&) {
        std::cerr << "error: allocation exceeded the enforced 4 GiB limit\n";
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
