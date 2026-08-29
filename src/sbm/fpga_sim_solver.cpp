#include "sbm/solver.hpp"

#include "../../fpga/dsb_hls.hpp"

#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace sbm {

SolverResult solve_fpga_sim(const BinaryQuadraticModel& bqm, const SolverParameters& p) {
    if (bqm.size() == 0 || bqm.size() > SBM_FPGA_MAX_SPINS) {
        throw std::invalid_argument("BQM size is outside the FPGA kernel's configured range");
    }
    if (p.steps <= 0 || p.runs <= 0) {
        throw std::invalid_argument("steps and runs must be positive");
    }
    const auto model = to_ising(bqm);
    const auto n = model.size();
    std::vector<sbm_fpga_value> dense(n * n, 0), fields(model.fields.begin(), model.fields.end());
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t edge = model.topology->row_offsets[row];
             edge < model.topology->row_offsets[row + 1]; ++edge) {
            dense[row * n + model.topology->columns[edge]] = model.couplings[edge];
        }
    }
    const auto c0 = static_cast<sbm_fpga_value>(p.c0 > 0.0 ? p.c0 : estimate_c0(model));
    SolverResult best;
    best.energy = std::numeric_limits<double>::infinity();

    for (int run = 0; run < p.runs; ++run) {
        std::mt19937_64 rng(
            p.seed + 0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(run));
        std::uniform_real_distribution<float> initial(
            -static_cast<float>(p.initial_scale), static_cast<float>(p.initial_scale));
        std::vector<sbm_fpga_value> x(n), y(n);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = initial(rng);
            y[i] = initial(rng);
        }
        dsb_hls(
            static_cast<int>(n), p.steps, dense.data(), fields.data(), x.data(), y.data(),
            static_cast<sbm_fpga_value>(p.a0), static_cast<sbm_fpga_value>(p.dt), c0,
            static_cast<sbm_fpga_value>(p.gamma));

        std::vector<std::uint8_t> sample(n);
        for (std::size_t i = 0; i < n; ++i) sample[i] = x[i] >= 0;
        const double energy = bqm.energy(sample);
        if (energy < best.energy) best = {std::move(sample), energy};
    }
    return best;
}

}  // namespace sbm
