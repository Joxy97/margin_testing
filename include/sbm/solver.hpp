#pragma once

#include "sbm/model.hpp"

#include <cstdint>
#include <vector>

namespace sbm {

struct SolverParameters {
    int steps = 10'000;
    int runs = 16;
    double dt = 1.0;
    double a0 = 1.0;
    double c0 = 0.0;       // <= 0 selects the paper's automatic estimate.
    double gamma = 0.0;    // 0 disables heating.
    double initial_scale = 0.05;
    std::uint64_t seed = 1;
};

struct SolverResult {
    std::vector<std::uint8_t> sample;
    double energy = 0.0;
};

[[nodiscard]] double estimate_c0(const IsingModel& model);
[[nodiscard]] SolverResult solve_cpu(
    const BinaryQuadraticModel& bqm, const SolverParameters& parameters = {});
[[nodiscard]] std::vector<SolverResult> solve_cpu_batch(
    const std::vector<BinaryQuadraticModel>& bqms,
    const SolverParameters& parameters = {});

}  // namespace sbm
