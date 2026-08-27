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

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __linux__
#include <sys/resource.h>
#endif

using Clock = std::chrono::steady_clock;

struct Work {
    int sbm_steps;
    int sbm_runs;
    int greedy_sweeps;
    int greedy_runs;
    int sa_sweeps;
    int sa_runs;
};

struct Measurement {
    std::string method;
    double wall_ms;
    std::vector<double> cuts;
    std::string configuration;
};

Work work_for(std::size_t n) {
    if (n <= 256) return {1'000, 4, 30, 4, 100, 4};
    if (n <= 1'024) return {1'000, 4, 30, 4, 80, 4};
    if (n <= 10'000) return {250, 2, 15, 2, 40, 2};
    return {32, 1, 8, 1, 16, 1};
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

double exact(const sbm::maxcut::Graph& graph) {
    auto adjacency = sbm::maxcut::make_adjacency(graph);
    std::vector<std::uint8_t> partition(graph.vertices, 0);
    double current = 0.0;
    double best = 0.0;
    const std::uint64_t states = 1ULL << (graph.vertices - 1);
    for (std::uint64_t state = 1; state < states; ++state) {
        const auto vertex = 1 + static_cast<std::size_t>(__builtin_ctzll(state));
        current += gain(vertex, partition, adjacency);
        partition[vertex] ^= 1;
        best = std::max(best, current);
    }
    return best;
}

double greedy(
    const sbm::maxcut::Graph& graph, int runs, int sweeps, std::uint64_t seed) {
    auto adjacency = sbm::maxcut::make_adjacency(graph);
    std::mt19937_64 rng(seed);
    std::vector<std::uint32_t> order(graph.vertices);
    std::iota(order.begin(), order.end(), 0);
    double best = 0.0;
    for (int run = 0; run < runs; ++run) {
        std::vector<std::uint8_t> partition(graph.vertices);
        for (auto& bit : partition) bit = rng() & 1U;
        double current = graph.cut_value(partition);
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            std::shuffle(order.begin(), order.end(), rng);
            bool changed = false;
            for (auto vertex : order) {
                const double delta = gain(vertex, partition, adjacency);
                if (delta > 0.0) {
                    partition[vertex] ^= 1;
                    current += delta;
                    changed = true;
                }
            }
            if (!changed) break;
        }
        best = std::max(best, current);
    }
    return best;
}

double anneal(
    const sbm::maxcut::Graph& graph, int runs, int sweeps, std::uint64_t seed) {
    auto adjacency = sbm::maxcut::make_adjacency(graph);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> probability(0.0, 1.0);
    std::uniform_int_distribution<std::uint32_t> vertex(
        0, static_cast<std::uint32_t>(graph.vertices - 1));
    const double total_weight = std::accumulate(
        graph.edges.begin(), graph.edges.end(), 0.0,
        [](double sum, const auto& edge) { return sum + edge.weight; });
    const double t0 = std::max(1.0, 4.0 * total_weight / graph.vertices);
    const double t1 = 0.01 * t0;
    double best = 0.0;
    for (int run = 0; run < runs; ++run) {
        std::vector<std::uint8_t> partition(graph.vertices);
        for (auto& bit : partition) bit = rng() & 1U;
        double current = graph.cut_value(partition);
        best = std::max(best, current);
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            const double fraction = sweeps == 1 ? 1.0 : static_cast<double>(sweep) / (sweeps - 1);
            const double temperature = t0 * std::pow(t1 / t0, fraction);
            for (std::size_t proposal = 0; proposal < graph.vertices; ++proposal) {
                const auto candidate = vertex(rng);
                const double delta = gain(candidate, partition, adjacency);
                if (delta >= 0.0 || probability(rng) < std::exp(delta / temperature)) {
                    partition[candidate] ^= 1;
                    current += delta;
                    best = std::max(best, current);
                }
            }
        }
    }
    return best;
}

template <class Function>
Measurement parallel_measure(
    const std::string& method, const std::vector<sbm::maxcut::Graph>& graphs,
    const std::string& configuration, Function&& function) {
    Measurement measurement{method, 0.0, std::vector<double>(graphs.size()), configuration};
    const auto start = Clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(graphs.size()); ++i) {
        measurement.cuts[i] = function(graphs[i], static_cast<std::size_t>(i));
    }
    measurement.wall_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return measurement;
}

int main(int argc, char** argv) {
    try {
        const std::filesystem::path output_path =
            argc > 1 ? argv[1] : "benchmarks/maxcut_batch/results.csv";
        const int batch_size = argc > 2 ? std::stoi(argv[2]) : 10;
        if (batch_size <= 0) throw std::invalid_argument("batch size must be positive");
        if (!output_path.parent_path().empty()) {
            std::filesystem::create_directories(output_path.parent_path());
        }
        std::ofstream output(output_path);
        if (!output) throw std::runtime_error("cannot write batch benchmark results");
        output << "vertices,edges_per_qubo,batch_size,method,batch_wall_ms,qubo_per_second,"
                  "mean_quality_ratio,minimum_quality_ratio,best_observed_wins,peak_rss_mib,"
                  "configuration\n";
        output << std::setprecision(17);

        const std::vector<std::size_t> sizes{20, 64, 256, 1'024, 10'000, 100'000};
        constexpr std::uint64_t base_seed = 0xba7c'2026ULL;
        for (auto vertices : sizes) {
            const std::size_t edges = std::min(vertices * 8, vertices * (vertices - 1) / 2);
            const auto work = work_for(vertices);
            std::cerr << "generating batch: n=" << vertices << ", count=" << batch_size << '\n';
            std::vector<sbm::maxcut::Graph> graphs(batch_size);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < batch_size; ++i) {
                graphs[i] = sbm::maxcut::generate(
                    vertices, edges, base_seed + vertices * 1'000 + i);
            }

            std::vector<Measurement> methods;
            if (vertices <= 20) {
                methods.push_back(parallel_measure(
                    "exact", graphs, "gray-code enumeration",
                    [](const auto& graph, std::size_t) { return exact(graph); }));
            }
            methods.push_back(parallel_measure(
                "greedy_local_search", graphs,
                "runs=" + std::to_string(work.greedy_runs) +
                    ";max_sweeps=" + std::to_string(work.greedy_sweeps),
                [&](const auto& graph, std::size_t i) {
                    return greedy(graph, work.greedy_runs, work.greedy_sweeps, base_seed + i);
                }));
            methods.push_back(parallel_measure(
                "simulated_annealing", graphs,
                "runs=" + std::to_string(work.sa_runs) +
                    ";sweeps=" + std::to_string(work.sa_sweeps),
                [&](const auto& graph, std::size_t i) {
                    return anneal(graph, work.sa_runs, work.sa_sweeps, base_seed + 100'000 + i);
                }));

            std::vector<sbm::BinaryQuadraticModel> bqms(batch_size);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < batch_size; ++i) bqms[i] = graphs[i].to_bqm();
            graphs.clear();
            graphs.shrink_to_fit();

            sbm::SolverParameters parameters;
            parameters.steps = work.sbm_steps;
            parameters.runs = work.sbm_runs;
            parameters.seed = base_seed + 200'000;
            const std::string sbm_configuration =
                "runs=" + std::to_string(work.sbm_runs) +
                ";steps=" + std::to_string(work.sbm_steps);

            Measurement sequential{
                "dsb_row_parallel", 0.0, std::vector<double>(batch_size),
                sbm_configuration + ";qubo_scheduling=sequential;simd=on"};
            auto start = Clock::now();
            for (int i = 0; i < batch_size; ++i) {
                auto local = parameters;
                local.seed += 0xd1b54a32d192ed03ULL * static_cast<std::uint64_t>(i);
                sequential.cuts[i] = -sbm::solve_cpu(bqms[i], local).energy;
            }
            sequential.wall_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            methods.push_back(std::move(sequential));

            start = Clock::now();
            auto batch_results = sbm::solve_cpu_batch(bqms, parameters);
            const double batch_wall =
                std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            Measurement parallel{
                "dsb_parallel_batch", batch_wall, std::vector<double>(batch_size),
                sbm_configuration + ";qubo_scheduling=adaptive;simd=on"};
            for (int i = 0; i < batch_size; ++i) parallel.cuts[i] = -batch_results[i].energy;
            methods.push_back(std::move(parallel));

            for (int i = 0; i < batch_size; ++i) {
                double reference = 0.0;
                for (const auto& method : methods) reference = std::max(reference, method.cuts[i]);
                for (auto& method : methods) method.cuts[i] /= reference;
            }
            const double memory = peak_rss_mib();
            for (const auto& method : methods) {
                const double mean = std::accumulate(method.cuts.begin(), method.cuts.end(), 0.0) /
                                    method.cuts.size();
                const double minimum = *std::min_element(method.cuts.begin(), method.cuts.end());
                const auto wins = std::count_if(
                    method.cuts.begin(), method.cuts.end(),
                    [](double ratio) { return ratio >= 1.0 - 1e-12; });
                const double throughput = 1'000.0 * batch_size / method.wall_ms;
                output << vertices << ',' << edges << ',' << batch_size << ',' << method.method
                       << ',' << method.wall_ms << ',' << throughput << ',' << mean << ','
                       << minimum << ',' << wins << ',' << memory << ',' << '"'
                       << method.configuration << '"' << '\n';
                std::cerr << "  " << method.method << ": " << throughput
                          << " QUBO/s, quality=" << mean << '\n';
            }
            output.flush();
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
