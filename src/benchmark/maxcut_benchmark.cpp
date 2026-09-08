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

MethodResult exact_maxcut(const sbm::maxcut::PreparedSearch& search) {
    const auto start = Clock::now();
    const auto result = search.exact();
    return {"exact", result.cut, std::chrono::duration<double, std::milli>(Clock::now() - start).count(),
            "gray-code enumeration;preparation=excluded"};
}

MethodResult greedy_maxcut(const sbm::maxcut::PreparedSearch& search,
    int runs, int sweeps, std::uint64_t seed) {
    const auto start = Clock::now();
    const auto result = search.greedy(runs, sweeps, seed);
    return {"greedy_local_search", result.cut, std::chrono::duration<double, std::milli>(Clock::now() - start).count(),
            "runs=" + std::to_string(runs) + ";max_sweeps=" + std::to_string(sweeps) + ";preparation=excluded"};
}

MethodResult simulated_annealing(const sbm::maxcut::PreparedSearch& search,
    int runs, int sweeps, std::uint64_t seed) {
    const auto start = Clock::now();
    const auto result = search.anneal(runs, sweeps, seed);
    return {"simulated_annealing", result.cut, std::chrono::duration<double, std::milli>(Clock::now() - start).count(),
            "runs=" + std::to_string(runs) + ";sweeps=" + std::to_string(sweeps) + ";preparation=excluded"};
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
            sbm::maxcut::PreparedSearch search(graph);
            const double load_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - load_start).count();
            const auto scale = scale_for(graph.vertices);
            const auto seed = 0x5eedULL + graph.vertices;
            std::cerr << "benchmarking " << path.filename().string() << " (n="
                      << graph.vertices << ", m=" << graph.edges.size() << ")\n";

            std::vector<MethodResult> methods;
            if (graph.vertices <= 24) methods.push_back(exact_maxcut(search));
            methods.push_back(greedy_maxcut(
                search, scale.greedy_runs, scale.greedy_sweeps, seed));
            methods.push_back(simulated_annealing(
                search, scale.sa_runs, scale.sa_sweeps, seed + 1));
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
