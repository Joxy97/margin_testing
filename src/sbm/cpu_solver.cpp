#include "sbm/solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sbm {
namespace {

void validate(const BinaryQuadraticModel& bqm, const SolverParameters& p) {
    if (bqm.size() == 0) throw std::invalid_argument("BQM must not be empty");
    if (p.steps <= 0 || p.runs <= 0) throw std::invalid_argument("steps and runs must be positive");
    if (!(p.dt > 0.0) || !(p.a0 > 0.0)) throw std::invalid_argument("dt and a0 must be positive");
    if (p.gamma < 0.0) throw std::invalid_argument("gamma must be non-negative");
}

std::vector<std::uint8_t> run_once(
    const IsingModel& model, const SolverParameters& p, double c0, std::uint64_t seed,
    bool parallel_rows) {
    const auto n = model.size();
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> initial(-p.initial_scale, p.initial_scale);
    std::vector<double> x(n), y(n), accum(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = initial(rng);
        y[i] = initial(rng);
    }

    const double da = p.a0 / p.steps;
    if (!parallel_rows) {
        for (int step = 0; step < p.steps; ++step) {
            const double a = step * da;
            for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(n); ++row) {
                double interaction = 0.0;
#ifdef _OPENMP
#pragma omp simd reduction(+ : interaction)
#endif
                for (std::ptrdiff_t edge = static_cast<std::ptrdiff_t>(model.row_offsets[row]);
                     edge < static_cast<std::ptrdiff_t>(model.row_offsets[row + 1]); ++edge) {
                    interaction += x[model.columns[edge]] >= 0.0
                                       ? model.couplings[edge]
                                       : -model.couplings[edge];
                }
                accum[row] = model.fields[row] + interaction;
            }
#ifdef _OPENMP
#pragma omp simd
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                const double previous_y = y[i];
                y[i] += ((a - p.a0) * x[i] + c0 * accum[i]) * p.dt;
                x[i] += p.a0 * y[i] * p.dt;
                if (std::abs(x[i]) > 1.0) {
                    x[i] = x[i] >= 0.0 ? 1.0 : -1.0;
                    y[i] = 0.0;
                }
                if (p.gamma != 0.0) y[i] += p.gamma * previous_y * p.dt;
            }
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            for (int step = 0; step < p.steps; ++step) {
                const double a = step * da;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(n); ++row) {
                    double interaction = 0.0;
#ifdef _OPENMP
#pragma omp simd reduction(+ : interaction)
#endif
                    for (std::ptrdiff_t edge =
                             static_cast<std::ptrdiff_t>(model.row_offsets[row]);
                         edge < static_cast<std::ptrdiff_t>(model.row_offsets[row + 1]); ++edge) {
                        interaction += x[model.columns[edge]] >= 0.0
                                           ? model.couplings[edge]
                                           : -model.couplings[edge];
                    }
                    accum[row] = model.fields[row] + interaction;
                }

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                    const double previous_y = y[i];
                    y[i] += ((a - p.a0) * x[i] + c0 * accum[i]) * p.dt;
                    x[i] += p.a0 * y[i] * p.dt;
                    if (std::abs(x[i]) > 1.0) {
                        x[i] = x[i] >= 0.0 ? 1.0 : -1.0;
                        y[i] = 0.0;
                    }
                    if (p.gamma != 0.0) y[i] += p.gamma * previous_y * p.dt;
                }
            }
        }
    }

    std::vector<std::uint8_t> sample(n);
    for (std::size_t i = 0; i < n; ++i) sample[i] = x[i] >= 0.0;
    return sample;
}

}  // namespace

double estimate_c0(const IsingModel& model) {
    const double matrix_n = static_cast<double>(model.size() + 1);  // includes ancillary spin
    double sum_of_squares = 0.0;
    for (double j : model.couplings) sum_of_squares += j * j;
    for (double h : model.fields) sum_of_squares += 2.0 * h * h;
    const double sigma = std::sqrt(sum_of_squares / (matrix_n * (matrix_n - 1.0)));
    return sigma > 0.0 ? 0.5 / (sigma * std::sqrt(matrix_n)) : 1.0;
}

SolverResult solve_cpu(const BinaryQuadraticModel& bqm, const SolverParameters& parameters) {
    validate(bqm, parameters);
    const auto model = to_ising(bqm);
    const double c0 = parameters.c0 > 0.0 ? parameters.c0 : estimate_c0(model);

    SolverResult best;
    best.energy = std::numeric_limits<double>::infinity();
    std::vector<SolverResult> candidates(parameters.runs);
#ifdef _OPENMP
    const bool inside_parallel_region = omp_in_parallel();
    const bool parallel_runs = !inside_parallel_region && parameters.runs > 1 && model.size() < 4'096;
#pragma omp parallel for if(parallel_runs) schedule(static)
#else
    const bool inside_parallel_region = false;
    const bool parallel_runs = false;
#endif
    for (int run = 0; run < parameters.runs; ++run) {
        auto sample = run_once(
            model, parameters, c0,
            parameters.seed + 0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(run),
            !inside_parallel_region && !parallel_runs);
        candidates[run] = {std::move(sample), 0.0};
        candidates[run].energy = bqm.energy(candidates[run].sample);
    }
    for (auto& candidate : candidates) {
        if (candidate.energy < best.energy) best = std::move(candidate);
    }
    return best;
}

std::vector<SolverResult> solve_cpu_batch(
    const std::vector<BinaryQuadraticModel>& bqms, const SolverParameters& parameters) {
    if (bqms.empty()) return {};
    std::vector<SolverResult> results(bqms.size());
    const auto largest_model = std::max_element(
        bqms.begin(), bqms.end(),
        [](const auto& left, const auto& right) { return left.size() < right.size(); });
    const bool parallelize_qubos = largest_model->size() < 20'000;

    if (parallelize_qubos) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(bqms.size()); ++i) {
            auto local_parameters = parameters;
            local_parameters.seed +=
                0xd1b54a32d192ed03ULL * static_cast<std::uint64_t>(i);
            results[i] = solve_cpu(bqms[i], local_parameters);
        }
    } else {
        for (std::size_t i = 0; i < bqms.size(); ++i) {
            auto local_parameters = parameters;
            local_parameters.seed += 0xd1b54a32d192ed03ULL * i;
            results[i] = solve_cpu(bqms[i], local_parameters);
        }
    }
    return results;
}

}  // namespace sbm
