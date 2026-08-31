#include "sbm/solver.hpp"

#include <algorithm>
#include <cmath>
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

void validate(const IsingModel& model, const SolverParameters& p) {
    if (model.size() == 0) throw std::invalid_argument("Ising model must not be empty");
    if (model.topology == nullptr ||
        model.topology->row_offsets.size() != model.size() + 1 ||
        model.topology->columns.size() != model.couplings.size()) {
        throw std::invalid_argument("invalid Ising topology");
    }
    if (p.steps <= 0 || p.runs <= 0) throw std::invalid_argument("steps and runs must be positive");
    if (!(p.dt > 0.0) || !(p.a0 > 0.0)) throw std::invalid_argument("dt and a0 must be positive");
    if (p.gamma < 0.0) throw std::invalid_argument("gamma must be non-negative");
}

double binary_energy(
    const IsingModel& model, const std::vector<std::uint8_t>& sample) {
    double value = model.offset;
    for (std::size_t row = 0; row < model.size(); ++row) {
        const double spin = sample[row] != 0 ? 1.0 : -1.0;
        value -= model.fields[row] * spin;
        for (std::size_t edge = model.topology->row_offsets[row];
             edge < model.topology->row_offsets[row + 1]; ++edge) {
            const double neighbor =
                sample[model.topology->columns[edge]] != 0 ? 1.0 : -1.0;
            value -= 0.5 * model.couplings[edge] * spin * neighbor;
        }
    }
    return value;
}

struct TrajectoryWorkspace {
    std::vector<float> positions;
    std::vector<float> momenta;
    std::vector<float> interactions;
    std::vector<std::int8_t> spins;

    void resize(std::size_t size) {
        positions.resize(size);
        momenta.resize(size);
        interactions.resize(size);
        spins.resize(size);
    }
};

std::vector<std::uint8_t> run_once(
    const IsingModel& model, const SolverParameters& p, float c0, std::uint64_t seed,
    bool parallel_rows, const std::vector<std::uint8_t>* initial_sample = nullptr) {
    const auto n = model.size();
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> initial(-p.initial_scale, p.initial_scale);

    // A worker repeatedly solves models of similar sizes. Retaining its buffers
    // avoids allocating four trajectory-sized arrays for every randomized run.
    thread_local TrajectoryWorkspace workspace;
    workspace.resize(n);
    auto& x = workspace.positions;
    auto& y = workspace.momenta;
    auto& accum = workspace.interactions;
    auto& spins = workspace.spins;
    for (std::size_t i = 0; i < n; ++i) {
        if (initial_sample != nullptr) {
            x[i] = (*initial_sample)[i] != 0 ? 1.0F : -1.0F;
        } else {
            x[i] = initial(rng);
        }
        y[i] = initial(rng);
        spins[i] = x[i] >= 0.0F ? 1 : -1;
    }

    const float da = p.a0 / static_cast<float>(p.steps);
    const float position_scale = p.a0 * p.dt;
    const float coupling_scale = c0 * p.dt;
    const float heating_scale = p.gamma * p.dt;
    if (!parallel_rows) {
        for (int step = 0; step < p.steps; ++step) {
            const float pump_scale = (static_cast<float>(step) * da - p.a0) * p.dt;
            for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(n); ++row) {
                float interaction = 0.0F;
#ifdef _OPENMP
#pragma omp simd reduction(+ : interaction)
#endif
                for (std::ptrdiff_t edge = static_cast<std::ptrdiff_t>(model.topology->row_offsets[row]);
                     edge < static_cast<std::ptrdiff_t>(model.topology->row_offsets[row + 1]); ++edge) {
                    interaction += model.couplings[edge] *
                                   spins[model.topology->columns[edge]];
                }
                accum[row] = model.fields[row] + interaction;
            }
#ifdef _OPENMP
#pragma omp simd
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                const float previous_y = y[i];
                y[i] += pump_scale * x[i] + coupling_scale * accum[i];
                x[i] += position_scale * y[i];
                if (std::abs(x[i]) > 1.0F) {
                    x[i] = x[i] >= 0.0F ? 1.0F : -1.0F;
                    y[i] = 0.0F;
                }
                if (heating_scale != 0.0F) y[i] += heating_scale * previous_y;
                spins[i] = x[i] >= 0.0F ? 1 : -1;
            }
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            for (int step = 0; step < p.steps; ++step) {
                const float pump_scale = (static_cast<float>(step) * da - p.a0) * p.dt;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(n); ++row) {
                    float interaction = 0.0F;
#ifdef _OPENMP
#pragma omp simd reduction(+ : interaction)
#endif
                    for (std::ptrdiff_t edge =
                             static_cast<std::ptrdiff_t>(model.topology->row_offsets[row]);
                         edge < static_cast<std::ptrdiff_t>(model.topology->row_offsets[row + 1]); ++edge) {
                        interaction += model.couplings[edge] *
                                       spins[model.topology->columns[edge]];
                    }
                    accum[row] = model.fields[row] + interaction;
                }

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                    const float previous_y = y[i];
                    y[i] += pump_scale * x[i] + coupling_scale * accum[i];
                    x[i] += position_scale * y[i];
                    if (std::abs(x[i]) > 1.0F) {
                        x[i] = x[i] >= 0.0F ? 1.0F : -1.0F;
                        y[i] = 0.0F;
                    }
                    if (heating_scale != 0.0F) y[i] += heating_scale * previous_y;
                    spins[i] = x[i] >= 0.0F ? 1 : -1;
                }
            }
        }
    }

    std::vector<std::uint8_t> sample(n);
    for (std::size_t i = 0; i < n; ++i) sample[i] = spins[i] > 0;
    return sample;
}

}  // namespace

float estimate_c0(const IsingModel& model) {
    const float matrix_n = static_cast<float>(model.size() + 1);  // includes ancillary spin
    float sum_of_squares = 0.0F;
    for (float j : model.couplings) sum_of_squares += j * j;
    for (float h : model.fields) sum_of_squares += 2.0F * h * h;
    const float sigma = std::sqrt(sum_of_squares / (matrix_n * (matrix_n - 1.0F)));
    return sigma > 0.0F ? 0.5F / (sigma * std::sqrt(matrix_n)) : 1.0F;
}

std::vector<SolverResult> solve_cpu_candidates(
    const BinaryQuadraticModel& bqm, const SolverParameters& parameters) {
    validate(bqm, parameters);
    const auto model = to_ising(bqm);
    auto candidates = solve_cpu_ising_candidates(model, parameters);
    for (auto& candidate : candidates) {
        candidate.energy = bqm.energy(candidate.sample);
    }
    return candidates;
}

std::vector<SolverResult> solve_cpu_ising_candidates(
    const IsingModel& model, const SolverParameters& parameters,
    const std::vector<std::uint8_t>& initial_sample) {
    validate(model, parameters);
    if (!initial_sample.empty() && initial_sample.size() != model.size()) {
        throw std::invalid_argument("warm-start sample size does not match model");
    }
    const float c0 = parameters.c0 > 0.0F ? parameters.c0 : estimate_c0(model);

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
        const auto* warm_start = run == 0 && !initial_sample.empty()
                                     ? &initial_sample
                                     : nullptr;
        auto sample = run_once(
            model, parameters, c0,
            parameters.seed + 0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(run),
            !inside_parallel_region && !parallel_runs, warm_start);
        candidates[run] = {std::move(sample), 0.0};
        candidates[run].energy = binary_energy(model, candidates[run].sample);
    }
    return candidates;
}

SolverResult solve_cpu(const BinaryQuadraticModel& bqm, const SolverParameters& parameters) {
    auto candidates = solve_cpu_candidates(bqm, parameters);
    auto best = std::min_element(
        candidates.begin(), candidates.end(),
        [](const auto& left, const auto& right) {
            return left.energy < right.energy;
        });
    return std::move(*best);
}

std::vector<SolverResult> solve_cpu_batch(
    const std::vector<BinaryQuadraticModel>& bqms, const SolverParameters& parameters) {
    auto candidate_batches = solve_cpu_candidates_batch(bqms, parameters);
    std::vector<SolverResult> results;
    results.reserve(candidate_batches.size());
    for (auto& candidates : candidate_batches) {
        auto best = std::min_element(
            candidates.begin(), candidates.end(),
            [](const auto& left, const auto& right) {
                return left.energy < right.energy;
            });
        results.push_back(std::move(*best));
    }
    return results;
}

std::vector<std::vector<SolverResult>> solve_cpu_candidates_batch(
    const std::vector<BinaryQuadraticModel>& bqms, const SolverParameters& parameters) {
    std::vector<IsingModel> models;
    models.reserve(bqms.size());
    for (const auto& bqm : bqms) models.push_back(to_ising(bqm));
    auto results = solve_cpu_ising_candidates_batch(models, parameters);
    for (std::size_t problem = 0; problem < results.size(); ++problem) {
        for (auto& candidate : results[problem]) {
            candidate.energy = bqms[problem].energy(candidate.sample);
        }
    }
    return results;
}

std::vector<std::vector<SolverResult>> solve_cpu_ising_candidates_batch(
    const std::vector<IsingModel>& models, const SolverParameters& parameters,
    const std::vector<std::vector<std::uint8_t>>& initial_samples,
    const std::vector<std::uint64_t>& problem_seeds) {
    if (models.empty()) return {};
    if (!initial_samples.empty() && initial_samples.size() != models.size()) {
        throw std::invalid_argument("warm-start batch size does not match models");
    }
    if (!problem_seeds.empty() && problem_seeds.size() != models.size()) {
        throw std::invalid_argument("problem-seed batch size does not match models");
    }
    std::vector<std::vector<SolverResult>> results(models.size());
    static const std::vector<std::uint8_t> empty_initial;
    const auto largest_model = std::max_element(
        models.begin(), models.end(),
        [](const auto& left, const auto& right) { return left.size() < right.size(); });
    const bool parallelize_qubos = largest_model->size() < 20'000;

    if (parallelize_qubos) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(models.size()); ++i) {
            auto local_parameters = parameters;
            local_parameters.seed = problem_seeds.empty()
                                        ? parameters.seed
                                              + 0xd1b54a32d192ed03ULL
                                                    * static_cast<std::uint64_t>(i)
                                        : problem_seeds[i];
            const auto& initial = initial_samples.empty()
                                      ? empty_initial
                                      : initial_samples[i];
            results[i] = solve_cpu_ising_candidates(
                models[i], local_parameters, initial);
        }
    } else {
        for (std::size_t i = 0; i < models.size(); ++i) {
            auto local_parameters = parameters;
            local_parameters.seed = problem_seeds.empty()
                                        ? parameters.seed
                                              + 0xd1b54a32d192ed03ULL * i
                                        : problem_seeds[i];
            const auto& initial = initial_samples.empty()
                                      ? empty_initial
                                      : initial_samples[i];
            results[i] = solve_cpu_ising_candidates(
                models[i], local_parameters, initial);
        }
    }
    return results;
}

}  // namespace sbm
