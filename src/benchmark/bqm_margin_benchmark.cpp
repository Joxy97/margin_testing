#include "sbm/model.hpp"
#include "sbm/solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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

namespace {

using Clock = std::chrono::steady_clock;

struct Input {
    sbm::BinaryQuadraticModel bqm;
    std::vector<double> portfolio_linear;
    std::vector<std::uint64_t> group_offsets;
};

struct Adjacency {
    std::vector<std::size_t> offsets;
    std::vector<std::uint32_t> neighbors;
    std::vector<double> biases;
};

struct Result {
    std::string method;
    std::vector<std::uint8_t> sample;
    double energy = 0.0;
    double milliseconds = 0.0;
    std::string configuration;
};

template <class T>
void read_exact(std::ifstream& input, T* destination, std::size_t count) {
    input.read(reinterpret_cast<char*>(destination),
               static_cast<std::streamsize>(sizeof(T) * count));
    if (!input) throw std::runtime_error("truncated compact BQM input");
}

Input load(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open compact BQM: " + path);
    char magic[8];
    read_exact(input, magic, 8);
    if (std::string(magic, 7) != "SBMBQM1") {
        throw std::runtime_error("invalid compact BQM magic");
    }

    std::uint64_t n = 0, m = 0, groups = 0;
    double offset = 0.0;
    read_exact(input, &n, 1);
    read_exact(input, &m, 1);
    read_exact(input, &groups, 1);
    read_exact(input, &offset, 1);

    Input result;
    result.bqm.offset = offset;
    result.bqm.linear.resize(n);
    result.portfolio_linear.resize(n);
    std::vector<std::uint32_t> heads(m), tails(m);
    std::vector<double> biases(m);
    result.group_offsets.resize(groups + 1);
    read_exact(input, result.bqm.linear.data(), n);
    read_exact(input, result.portfolio_linear.data(), n);
    read_exact(input, heads.data(), m);
    read_exact(input, tails.data(), m);
    read_exact(input, biases.data(), m);
    read_exact(input, result.group_offsets.data(), groups + 1);
    result.bqm.quadratic.reserve(m);
    for (std::size_t edge = 0; edge < m; ++edge) {
        result.bqm.quadratic.push_back({heads[edge], tails[edge], biases[edge]});
    }
    if (result.group_offsets.empty() || result.group_offsets.front() != 0 ||
        result.group_offsets.back() != n) {
        throw std::runtime_error("invalid one-hot group offsets");
    }
    return result;
}

Adjacency make_adjacency(const sbm::BinaryQuadraticModel& bqm) {
    Adjacency adjacency;
    adjacency.offsets.assign(bqm.size() + 1, 0);
    for (const auto& term : bqm.quadratic) {
        ++adjacency.offsets[term.u + 1];
        ++adjacency.offsets[term.v + 1];
    }
    std::partial_sum(adjacency.offsets.begin(), adjacency.offsets.end(),
                     adjacency.offsets.begin());
    adjacency.neighbors.resize(adjacency.offsets.back());
    adjacency.biases.resize(adjacency.offsets.back());
    auto cursor = adjacency.offsets;
    for (const auto& term : bqm.quadratic) {
        const auto uv = cursor[term.u]++;
        const auto vu = cursor[term.v]++;
        adjacency.neighbors[uv] = term.v;
        adjacency.biases[uv] = term.bias;
        adjacency.neighbors[vu] = term.u;
        adjacency.biases[vu] = term.bias;
    }
    return adjacency;
}

std::vector<double> local_fields(
    const sbm::BinaryQuadraticModel& bqm, const Adjacency& adjacency,
    const std::vector<std::uint8_t>& sample) {
    auto fields = bqm.linear;
    for (std::size_t i = 0; i < bqm.size(); ++i) {
        for (std::size_t p = adjacency.offsets[i]; p < adjacency.offsets[i + 1]; ++p) {
            fields[i] += adjacency.biases[p] * sample[adjacency.neighbors[p]];
        }
    }
    return fields;
}

void flip(
    std::size_t variable, std::vector<std::uint8_t>& sample,
    std::vector<double>& fields, const Adjacency& adjacency) {
    const int change = sample[variable] ? -1 : 1;
    sample[variable] ^= 1U;
    for (std::size_t p = adjacency.offsets[variable];
         p < adjacency.offsets[variable + 1]; ++p) {
        fields[adjacency.neighbors[p]] += adjacency.biases[p] * change;
    }
}

Result greedy(
    const sbm::BinaryQuadraticModel& bqm, int sweeps, std::uint64_t seed) {
    const auto start = Clock::now();
    const auto adjacency = make_adjacency(bqm);
    std::mt19937_64 rng(seed);
    std::vector<std::uint8_t> sample(bqm.size());
    for (auto& bit : sample) bit = static_cast<std::uint8_t>(rng() & 1U);
    auto fields = local_fields(bqm, adjacency, sample);
    std::vector<std::uint32_t> order(bqm.size());
    std::iota(order.begin(), order.end(), 0);
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        std::shuffle(order.begin(), order.end(), rng);
        bool improved = false;
        for (const auto variable : order) {
            const double delta = (sample[variable] ? -1.0 : 1.0) * fields[variable];
            if (delta < -1e-14) {
                flip(variable, sample, fields, adjacency);
                improved = true;
            }
        }
        if (!improved) break;
    }
    return {"greedy_local_search", std::move(sample), 0.0,
            std::chrono::duration<double, std::milli>(Clock::now() - start).count(),
            "runs=1;max_sweeps=" + std::to_string(sweeps)};
}

Result anneal(
    const sbm::BinaryQuadraticModel& bqm, int sweeps, std::uint64_t seed) {
    const auto start = Clock::now();
    const auto adjacency = make_adjacency(bqm);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> probability(0.0, 1.0);
    std::uniform_int_distribution<std::uint32_t> variable(
        0, static_cast<std::uint32_t>(bqm.size() - 1));
    std::vector<std::uint8_t> sample(bqm.size());
    for (auto& bit : sample) bit = static_cast<std::uint8_t>(rng() & 1U);
    auto fields = local_fields(bqm, adjacency, sample);
    double current = bqm.energy(sample);
    double best_energy = current;
    auto best_sample = sample;

    double scale = 0.0;
    for (double field : fields) scale += std::abs(field);
    scale = std::max(scale / static_cast<double>(fields.size()), 1e-9);
    const double initial_temperature = 2.0 * scale;
    const double final_temperature = 0.01 * scale;
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        const double fraction = sweeps == 1 ? 1.0 : static_cast<double>(sweep) / (sweeps - 1);
        const double temperature = initial_temperature *
            std::pow(final_temperature / initial_temperature, fraction);
        for (std::size_t proposal = 0; proposal < bqm.size(); ++proposal) {
            const auto candidate = variable(rng);
            const double delta = (sample[candidate] ? -1.0 : 1.0) * fields[candidate];
            if (delta <= 0.0 || probability(rng) < std::exp(-delta / temperature)) {
                flip(candidate, sample, fields, adjacency);
                current += delta;
                if (current < best_energy) {
                    best_energy = current;
                    best_sample = sample;
                }
            }
        }
    }
    return {"simulated_annealing", std::move(best_sample), best_energy,
            std::chrono::duration<double, std::milli>(Clock::now() - start).count(),
            "runs=1;sweeps=" + std::to_string(sweeps)};
}

Result dsb(const sbm::BinaryQuadraticModel& bqm, int steps, std::uint64_t seed) {
    sbm::SolverParameters parameters;
    parameters.steps = steps;
    parameters.runs = 1;
    parameters.seed = seed;
    const auto start = Clock::now();
    auto solved = sbm::solve_cpu(bqm, parameters);
    return {"simulated_bifurcation", std::move(solved.sample), solved.energy,
            std::chrono::duration<double, std::milli>(Clock::now() - start).count(),
            "runs=1;steps=" + std::to_string(steps) + ";openmp_rows=on;simd=on"};
}

double peak_rss_mib() {
#ifdef __linux__
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) return usage.ru_maxrss / 1024.0;
#endif
    return 0.0;
}

std::size_t violation_count(
    const std::vector<std::uint8_t>& sample,
    const std::vector<std::uint64_t>& group_offsets) {
    std::size_t violations = 0;
    for (std::size_t group = 0; group + 1 < group_offsets.size(); ++group) {
        std::size_t count = 0;
        for (std::size_t i = group_offsets[group]; i < group_offsets[group + 1]; ++i) {
            count += sample[i] != 0;
        }
        violations += count != 1;
    }
    return violations;
}

void repair_one_hot(std::vector<std::uint8_t>& sample, const Input& input) {
    if (violation_count(sample, input.group_offsets) == 0) return;
    const auto adjacency = make_adjacency(input.bqm);
    auto fields = local_fields(input.bqm, adjacency, sample);
    for (std::size_t group = 0; group + 1 < input.group_offsets.size(); ++group) {
        const auto begin = input.group_offsets[group];
        const auto end = input.group_offsets[group + 1];
        std::size_t count = 0;
        for (std::size_t i = begin; i < end; ++i) count += sample[i] != 0;
        if (count == 1) continue;
        for (std::size_t i = begin; i < end; ++i) {
            if (sample[i]) flip(i, sample, fields, adjacency);
        }
        std::size_t best = begin;
        for (std::size_t i = begin + 1; i < end; ++i) {
            if (fields[i] < fields[best]) best = i;
        }
        flip(best, sample, fields, adjacency);
    }
}

void print(const Result& result, const Input& input) {
    const std::size_t raw_violations = violation_count(result.sample, input.group_offsets);
    auto sample = result.sample;
    const auto repair_start = Clock::now();
    repair_one_hot(sample, input);
    const double repair_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - repair_start).count();
    const double energy = input.bqm.energy(sample);
    double portfolio_return = 0.0;
    std::size_t selected = 0;
    for (std::size_t i = 0; i < sample.size(); ++i) {
        if (sample[i]) {
            portfolio_return += input.portfolio_linear[i];
            ++selected;
        }
    }
    std::cout << result.method << '\t' << result.milliseconds << '\t' << energy << '\t'
              << portfolio_return << '\t' << -portfolio_return << '\t' << raw_violations << '\t'
              << repair_ms << '\t' << selected << '\t' << peak_rss_mib() << '\t'
              << result.configuration << ";one_hot_decode=on\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0]
                      << " COMPACT.bqm [steps=15] [greedy_sweeps=5] [sa_sweeps=8] [seed=1]\n";
            return 2;
        }
        const int steps = argc > 2 ? std::stoi(argv[2]) : 15;
        const int greedy_sweeps = argc > 3 ? std::stoi(argv[3]) : 5;
        const int sa_sweeps = argc > 4 ? std::stoi(argv[4]) : 8;
        const std::uint64_t seed = argc > 5 ? std::stoull(argv[5]) : 1;
        const auto input = load(argv[1]);
        std::cout << std::setprecision(17);
        print(greedy(input.bqm, greedy_sweeps, seed), input);
        print(anneal(input.bqm, sa_sweeps, seed + 1), input);
        print(dsb(input.bqm, steps, seed + 2), input);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
