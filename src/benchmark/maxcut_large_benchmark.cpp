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

Result greedy(const sbm::maxcut::Graph& graph, int sweeps, std::uint64_t seed) {
    const auto start = Clock::now();
    const auto result = sbm::maxcut::PreparedSearch(graph).greedy(1, sweeps, seed);
    return {result.cut, std::chrono::duration<double, std::milli>(Clock::now() - start).count()};
}

Result anneal(const sbm::maxcut::Graph& graph, int sweeps, std::uint64_t seed) {
    const auto start = Clock::now();
    const auto result = sbm::maxcut::PreparedSearch(graph).anneal(1, sweeps, seed);
    return {result.cut, std::chrono::duration<double, std::milli>(Clock::now() - start).count()};
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
                      "runs=1;max_sweeps=" + std::to_string(greedy_sweeps) + ";preparation=included");
                write("simulated_annealing", sa_result,
                      "runs=1;sweeps=" + std::to_string(sa_sweeps) + ";preparation=included");
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
