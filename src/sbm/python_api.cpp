#include "sbm/model.hpp"
#include "sbm/solver.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>
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
    const double* linear,
    std::size_t quadratic_count,
    const std::uint32_t* quadratic_u,
    const std::uint32_t* quadratic_v,
    const double* quadratic_bias,
    double offset,
    int steps,
    int runs,
    double dt,
    double a0,
    double c0,
    double gamma,
    double initial_scale,
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
    const double* linear,
    std::size_t quadratic_count,
    const std::uint32_t* quadratic_u,
    const std::uint32_t* quadratic_v,
    const double* quadratic_bias,
    double offset,
    int steps,
    int runs,
    double dt,
    double a0,
    double c0,
    double gamma,
    double initial_scale,
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

extern "C" int sbm_solve_qubo_cpu_candidates_batch(
    std::size_t problem_count,
    const std::size_t* variable_offsets,
    const double* linear,
    const std::size_t* quadratic_offsets,
    const std::uint32_t* quadratic_u,
    const std::uint32_t* quadratic_v,
    const double* quadratic_bias,
    const double* offsets,
    int steps,
    int runs,
    double dt,
    double a0,
    double c0,
    double gamma,
    double initial_scale,
    std::uint64_t seed,
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

        std::vector<sbm::BinaryQuadraticModel> bqms(problem_count);
        for (std::size_t problem = 0; problem < problem_count; ++problem) {
            const auto variable_start = variable_offsets[problem];
            const auto variable_stop = variable_offsets[problem + 1];
            const auto edge_start = quadratic_offsets[problem];
            const auto edge_stop = quadratic_offsets[problem + 1];
            auto& bqm = bqms[problem];
            bqm.linear.assign(linear + variable_start, linear + variable_stop);
            bqm.offset = offsets[problem];
            bqm.quadratic.reserve(edge_stop - edge_start);
            for (std::size_t edge = edge_start; edge < edge_stop; ++edge) {
                bqm.quadratic.push_back({
                    quadratic_u[edge],
                    quadratic_v[edge],
                    quadratic_bias[edge],
                });
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

        const auto batches = sbm::solve_cpu_candidates_batch(bqms, parameters);
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
