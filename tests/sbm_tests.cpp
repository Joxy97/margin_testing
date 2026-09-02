#include "sbm/model.hpp"
#include "sbm/maxcut.hpp"
#include "sbm/solver.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

sbm::BinaryQuadraticModel example() {
    return {{-1.0, -1.0}, {{0, 1, 2.0}}, 0.0};
}

void test_qubo_ising_equivalence() {
    const sbm::BinaryQuadraticModel bqm{
        {-1.25, 0.75, 2.0}, {{0, 1, -2.0}, {1, 2, 1.5}}, 0.375};
    const auto ising = sbm::to_ising(bqm);
    for (unsigned mask = 0; mask < 8; ++mask) {
        std::vector<std::uint8_t> sample{static_cast<std::uint8_t>(mask & 1),
                                         static_cast<std::uint8_t>((mask >> 1) & 1),
                                         static_cast<std::uint8_t>((mask >> 2) & 1)};
        std::vector<std::int8_t> spins{static_cast<std::int8_t>(2 * sample[0] - 1),
                                       static_cast<std::int8_t>(2 * sample[1] - 1),
                                       static_cast<std::int8_t>(2 * sample[2] - 1)};
        require(std::abs(bqm.energy(sample) - ising.energy(spins)) < 1e-12,
                "QUBO-to-Ising conversion changed the energy");
    }
}

void test_cpu_solver() {
    sbm::SolverParameters parameters;
    parameters.steps = 200;
    parameters.runs = 16;
    parameters.seed = 7;
    const auto result = sbm::solve_cpu(example(), parameters);
    require(std::abs(result.energy + 1.0) < 1e-12, "CPU solver missed the simple optimum");
}

void test_cpu_batch_solver() {
    const std::vector<sbm::BinaryQuadraticModel> bqms{example(), example()};
    sbm::SolverParameters parameters;
    parameters.steps = 100;
    parameters.runs = 2;
    parameters.seed = 11;
    const auto batch = sbm::solve_cpu_batch(bqms, parameters);
    require(batch.size() == bqms.size(), "CPU batch solver returned the wrong result count");
    for (std::size_t i = 0; i < bqms.size(); ++i) {
        auto local = parameters;
        local.seed += 0xd1b54a32d192ed03ULL * i;
        const auto single = sbm::solve_cpu(bqms[i], local);
        require(std::abs(batch[i].energy - single.energy) < 1e-12,
                "CPU batch and single-problem solvers disagree");
    }
}

void test_cpu_candidate_batch_solver() {
    const std::vector<sbm::BinaryQuadraticModel> bqms{example(), example()};
    sbm::SolverParameters parameters;
    parameters.steps = 100;
    parameters.runs = 3;
    parameters.seed = 13;

    const auto batches = sbm::solve_cpu_candidates_batch(bqms, parameters);

    require(batches.size() == bqms.size(),
            "CPU candidate batch returned the wrong problem count");
    for (std::size_t problem = 0; problem < batches.size(); ++problem) {
        require(batches[problem].size() ==
                    static_cast<std::size_t>(parameters.runs),
                "CPU candidate batch returned the wrong run count");
        for (const auto& candidate : batches[problem]) {
            require(candidate.sample.size() == bqms[problem].size(),
                    "CPU candidate batch returned the wrong sample size");
            require(std::abs(candidate.energy -
                             bqms[problem].energy(candidate.sample)) < 1e-12,
                    "CPU candidate batch returned an inconsistent energy");
        }
    }
}

void test_cpu_candidate_batch_uses_explicit_problem_seeds() {
    const std::vector<sbm::IsingModel> models{
        sbm::to_ising(example()), sbm::to_ising(example())};
    const std::vector<std::uint64_t> seeds{101, 202};
    sbm::SolverParameters parameters;
    parameters.steps = 40;
    parameters.runs = 3;

    const auto batches = sbm::solve_cpu_ising_candidates_batch(
        models, parameters, {}, seeds);

    for (std::size_t problem = 0; problem < models.size(); ++problem) {
        auto local = parameters;
        local.seed = seeds[problem];
        const auto single = sbm::solve_cpu_ising_candidates(
            models[problem], local);
        require(batches[problem].size() == single.size(),
                "explicit problem seeds changed candidate count");
        for (std::size_t run = 0; run < single.size(); ++run) {
            require(batches[problem][run].sample == single[run].sample,
                    "batch position changed an explicit problem seed");
            require(std::abs(batches[problem][run].energy - single[run].energy)
                        < 1e-12,
                    "explicit problem seed changed candidate energy");
        }
    }
}

void test_maxcut_qubo_equivalence() {
    const sbm::maxcut::Graph graph{
        3, {{0, 1, 2.0}, {1, 2, 3.0}, {0, 2, 1.0}}};
    const auto bqm = graph.to_bqm();
    for (unsigned mask = 0; mask < 8; ++mask) {
        std::vector<std::uint8_t> partition{
            static_cast<std::uint8_t>(mask & 1),
            static_cast<std::uint8_t>((mask >> 1) & 1),
            static_cast<std::uint8_t>((mask >> 2) & 1)};
        require(std::abs(bqm.energy(partition) + graph.cut_value(partition)) < 1e-12,
                "MaxCut QUBO energy is not the negative cut value");
    }
}

#ifdef SBM_HAS_FPGA_SIM
void test_fpga_sim_solver() {
    sbm::SolverParameters parameters;
    parameters.steps = 200;
    parameters.runs = 16;
    parameters.seed = 7;
    const auto result = sbm::solve_fpga_sim(example(), parameters);
    require(std::abs(result.energy + 1.0) < 1e-12, "FPGA simulation missed the simple optimum");
}
#endif

}  // namespace

int main() {
    try {
        test_qubo_ising_equivalence();
        test_maxcut_qubo_equivalence();
        test_cpu_solver();
        test_cpu_batch_solver();
        test_cpu_candidate_batch_solver();
        test_cpu_candidate_batch_uses_explicit_problem_seeds();
#ifdef SBM_HAS_FPGA_SIM
        test_fpga_sim_solver();
#endif
        std::cout << "all tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
