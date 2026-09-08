#include "sbm/model.hpp"
#include "sbm/solver.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void copy_error(const char* message, char* output, std::size_t output_size) {
    if (output == nullptr || output_size == 0) return;
    std::strncpy(output, message, output_size - 1);
    output[output_size - 1] = '\0';
}

}  // namespace

extern "C" int sbm_solve_qubo_cpu(
    std::size_t variable_count,
    const float* linear,
    std::size_t quadratic_count,
    const std::uint32_t* quadratic_u,
    const std::uint32_t* quadratic_v,
    const float* quadratic_bias,
    float offset,
    int steps,
    int runs,
    float dt,
    float a0,
    float c0,
    float gamma,
    float initial_scale,
    std::uint64_t seed,
    std::uint8_t* sample,
    double* energy,
    char* error,
    std::size_t error_size) {
    try {
        if (linear == nullptr || sample == nullptr || energy == nullptr) {
            throw std::invalid_argument("null input or output buffer");
        }
        sbm::BinaryQuadraticModel bqm;
        bqm.linear.assign(linear, linear + variable_count);
        bqm.offset = offset;
        bqm.quadratic.reserve(quadratic_count);
        for (std::size_t edge = 0; edge < quadratic_count; ++edge) {
            bqm.quadratic.push_back({
                quadratic_u[edge],
                quadratic_v[edge],
                quadratic_bias[edge],
            });
        }

        sbm::SolverParameters parameters;
        parameters.steps = steps;
        parameters.runs = runs;
        parameters.dt = dt;
        parameters.a0 = a0;
        parameters.c0 = c0;
        parameters.gamma = gamma;
        parameters.initial_scale = initial_scale;
        parameters.seed = seed;

        const auto result = sbm::solve_cpu(bqm, parameters);
        std::copy(result.sample.begin(), result.sample.end(), sample);
        *energy = result.energy;
        if (error != nullptr && error_size > 0) error[0] = '\0';
        return 0;
    } catch (const std::exception& exception) {
        copy_error(exception.what(), error, error_size);
        return 1;
    } catch (...) {
        copy_error("unknown C++ exception", error, error_size);
        return 2;
    }
}

extern "C" int sbm_solve_qubo_cpu_candidates(
    std::size_t variable_count,
    const float* linear,
    std::size_t quadratic_count,
    const std::uint32_t* quadratic_u,
    const std::uint32_t* quadratic_v,
    const float* quadratic_bias,
    float offset,
    int steps,
    int runs,
    float dt,
    float a0,
    float c0,
    float gamma,
    float initial_scale,
    std::uint64_t seed,
    std::uint8_t* samples,
    double* energies,
    char* error,
    std::size_t error_size) {
    try {
        if (linear == nullptr || samples == nullptr || energies == nullptr) {
            throw std::invalid_argument("null input or output buffer");
        }
        sbm::BinaryQuadraticModel bqm;
        bqm.linear.assign(linear, linear + variable_count);
        bqm.offset = offset;
        bqm.quadratic.reserve(quadratic_count);
        for (std::size_t edge = 0; edge < quadratic_count; ++edge) {
            bqm.quadratic.push_back({
                quadratic_u[edge],
                quadratic_v[edge],
                quadratic_bias[edge],
            });
        }

        sbm::SolverParameters parameters;
        parameters.steps = steps;
        parameters.runs = runs;
        parameters.dt = dt;
        parameters.a0 = a0;
        parameters.c0 = c0;
        parameters.gamma = gamma;
        parameters.initial_scale = initial_scale;
        parameters.seed = seed;

        const auto candidates = sbm::solve_cpu_candidates(bqm, parameters);
        for (std::size_t run = 0; run < candidates.size(); ++run) {
            std::copy(
                candidates[run].sample.begin(),
                candidates[run].sample.end(),
                samples + run * variable_count);
            energies[run] = candidates[run].energy;
        }
        if (error != nullptr && error_size > 0) error[0] = '\0';
        return 0;
    } catch (const std::exception& exception) {
        copy_error(exception.what(), error, error_size);
        return 1;
    } catch (...) {
        copy_error("unknown C++ exception", error, error_size);
        return 2;
    }
}

extern "C" int sbm_solve_qubo_cpu_candidates_seeded_batch_prepared(
    void* preparation_context,
    std::size_t problem_count,
    const std::size_t* variable_offsets,
    const float* linear,
    const std::size_t* quadratic_offsets,
    const std::uint32_t* quadratic_u,
    const std::uint32_t* quadratic_v,
    const float* quadratic_bias,
    const float* offsets,
    int steps,
    int runs,
    float dt,
    float a0,
    float c0,
    float gamma,
    float initial_scale,
    std::uint64_t seed,
    const std::uint64_t* problem_seeds,
    const std::uint8_t* initial_samples,
    const std::uint8_t* initial_sample_flags,
    std::size_t topology_cache_bytes,
    std::uint8_t* samples,
    double* energies,
    char* error,
    std::size_t error_size) {
    try {
        if (problem_count == 0) return 0;
        if (variable_offsets == nullptr || linear == nullptr ||
            quadratic_offsets == nullptr || offsets == nullptr ||
            samples == nullptr || energies == nullptr) {
            throw std::invalid_argument("null batch input or output buffer");
        }
        if (quadratic_offsets[problem_count] != 0 &&
            (quadratic_u == nullptr || quadratic_v == nullptr ||
             quadratic_bias == nullptr)) {
            throw std::invalid_argument("null quadratic batch input");
        }

        sbm::QUBOPreparation local_preparation(topology_cache_bytes);
        auto& preparation = preparation_context
            ? *static_cast<sbm::QUBOPreparation*>(preparation_context) : local_preparation;
        preparation.set_memory_budget(topology_cache_bytes);
        std::vector<sbm::IsingModel> models;
        std::vector<std::vector<std::uint8_t>> warm_starts;
        models.reserve(problem_count);
        if (initial_sample_flags != nullptr) warm_starts.resize(problem_count);
        for (std::size_t problem = 0; problem < problem_count; ++problem) {
            const auto variable_start = variable_offsets[problem];
            const auto variable_stop = variable_offsets[problem + 1];
            const auto edge_start = quadratic_offsets[problem];
            const auto edge_stop = quadratic_offsets[problem + 1];
            if (variable_stop < variable_start || edge_stop < edge_start)
                throw std::invalid_argument("batch offsets must be nondecreasing");
            models.push_back(preparation.prepare(
                variable_stop - variable_start, linear + variable_start,
                edge_stop - edge_start,
                quadratic_u ? quadratic_u + edge_start : nullptr,
                quadratic_v ? quadratic_v + edge_start : nullptr,
                quadratic_bias ? quadratic_bias + edge_start : nullptr,
                offsets[problem]));
            if (initial_sample_flags != nullptr &&
                initial_sample_flags[problem] != 0) {
                if (initial_samples == nullptr) {
                    throw std::invalid_argument("warm-start flags require samples");
                }
                warm_starts[problem].assign(
                    initial_samples + variable_start,
                    initial_samples + variable_stop);
            }
        }

        sbm::SolverParameters parameters;
        parameters.steps = steps;
        parameters.runs = runs;
        parameters.dt = dt;
        parameters.a0 = a0;
        parameters.c0 = c0;
        parameters.gamma = gamma;
        parameters.initial_scale = initial_scale;
        parameters.seed = seed;

        std::vector<std::uint64_t> seeds;
        if (problem_seeds != nullptr) {
            seeds.assign(problem_seeds, problem_seeds + problem_count);
        }
        const auto batches = sbm::solve_cpu_ising_candidates_batch(
            models, parameters, warm_starts, seeds);
        std::size_t sample_cursor = 0;
        for (std::size_t problem = 0; problem < batches.size(); ++problem) {
            const auto variable_count =
                variable_offsets[problem + 1] - variable_offsets[problem];
            for (std::size_t run = 0; run < batches[problem].size(); ++run) {
                const auto& candidate = batches[problem][run];
                std::copy(
                    candidate.sample.begin(),
                    candidate.sample.end(),
                    samples + sample_cursor);
                sample_cursor += variable_count;
                energies[problem * static_cast<std::size_t>(runs) + run] =
                    candidate.energy;
            }
        }
        if (error != nullptr && error_size > 0) error[0] = '\0';
        return 0;
    } catch (const std::exception& exception) {
        copy_error(exception.what(), error, error_size);
        return 1;
    } catch (...) {
        copy_error("unknown C++ exception", error, error_size);
        return 2;
    }
}

extern "C" int sbm_solve_qubo_cpu_candidates_seeded_batch(
    std::size_t problem_count,
    const std::size_t* variable_offsets,
    const float* linear,
    const std::size_t* quadratic_offsets,
    const std::uint32_t* quadratic_u,
    const std::uint32_t* quadratic_v,
    const float* quadratic_bias,
    const float* offsets,
    int steps,
    int runs,
    float dt,
    float a0,
    float c0,
    float gamma,
    float initial_scale,
    std::uint64_t seed,
    const std::uint64_t* problem_seeds,
    const std::uint8_t* initial_samples,
    const std::uint8_t* initial_sample_flags,
    std::size_t topology_cache_bytes,
    std::uint8_t* samples,
    double* energies,
    char* error,
    std::size_t error_size) {
    return sbm_solve_qubo_cpu_candidates_seeded_batch_prepared(nullptr,
        problem_count, variable_offsets, linear, quadratic_offsets, quadratic_u, quadratic_v, quadratic_bias, offsets, steps, runs, dt, a0, c0, gamma, initial_scale, seed, problem_seeds, initial_samples, initial_sample_flags, topology_cache_bytes, samples, energies, error, error_size);
}

extern "C" int sbm_solve_qubo_cpu_candidates_batch(
    std::size_t problem_count,
    const std::size_t* variable_offsets,
    const float* linear,
    const std::size_t* quadratic_offsets,
    const std::uint32_t* quadratic_u,
    const std::uint32_t* quadratic_v,
    const float* quadratic_bias,
    const float* offsets,
    int steps,
    int runs,
    float dt,
    float a0,
    float c0,
    float gamma,
    float initial_scale,
    std::uint64_t seed,
    const std::uint8_t* initial_samples,
    const std::uint8_t* initial_sample_flags,
    std::size_t topology_cache_bytes,
    std::uint8_t* samples,
    double* energies,
    char* error,
    std::size_t error_size) {
    return sbm_solve_qubo_cpu_candidates_seeded_batch(
        problem_count, variable_offsets, linear, quadratic_offsets,
        quadratic_u, quadratic_v, quadratic_bias, offsets, steps, runs, dt,
        a0, c0, gamma, initial_scale, seed, nullptr, initial_samples,
        initial_sample_flags, topology_cache_bytes, samples, energies, error,
        error_size);
}

extern "C" void* sbm_create_preparation() {
    try { return new sbm::QUBOPreparation(); } catch (...) { return nullptr; }
}
extern "C" void sbm_destroy_preparation(void* context) {
    delete static_cast<sbm::QUBOPreparation*>(context);
}
