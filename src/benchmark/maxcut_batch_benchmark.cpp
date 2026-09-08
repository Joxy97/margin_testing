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

double exact(const sbm::maxcut::Graph& graph) {
    return sbm::maxcut::PreparedSearch(graph).exact().cut;
}

double greedy(const sbm::maxcut::Graph& graph, int runs, int sweeps, std::uint64_t seed) {
    return sbm::maxcut::PreparedSearch(graph).greedy(runs, sweeps, seed).cut;
}

double anneal(const sbm::maxcut::Graph& graph, int runs, int sweeps, std::uint64_t seed) {
    return sbm::maxcut::PreparedSearch(graph).anneal(runs, sweeps, seed).cut;
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
                    ";max_sweeps=" + std::to_string(work.greedy_sweeps) + ";preparation=included",
                [&](const auto& graph, std::size_t i) {
                    return greedy(graph, work.greedy_runs, work.greedy_sweeps, base_seed + i);
                }));
            methods.push_back(parallel_measure(
                "simulated_annealing", graphs,
                "runs=" + std::to_string(work.sa_runs) +
                    ";sweeps=" + std::to_string(work.sa_sweeps) + ";preparation=included",
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
