#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sbm {

struct QuadraticBias {
    std::uint32_t u;
    std::uint32_t v;
    double bias;
};

// E(q) = offset + sum_i linear[i] q_i + sum_(i<j) bias_ij q_i q_j.
struct BinaryQuadraticModel {
    std::vector<double> linear;
    std::vector<QuadraticBias> quadratic;
    double offset = 0.0;

    [[nodiscard]] std::size_t size() const noexcept { return linear.size(); }
    [[nodiscard]] double energy(const std::vector<std::uint8_t>& sample) const;
};

// Paper convention: H(s) = offset - 1/2 s^T J s - h^T s.
// J is symmetric and stored as full CSR (both directions, zero diagonal).
struct IsingTopology {
    std::vector<std::size_t> row_offsets;
    std::vector<std::uint32_t> columns;
};

struct IsingModel {
    std::shared_ptr<const IsingTopology> topology;
    std::vector<float> couplings;
    std::vector<float> fields;
    float offset = 0.0F;

    [[nodiscard]] std::size_t size() const noexcept { return fields.size(); }
    [[nodiscard]] double energy(const std::vector<std::int8_t>& spins) const;
};

class QUBOPreparation {
public:
    explicit QUBOPreparation(std::size_t maximum_bytes = 0);
    ~QUBOPreparation();
    QUBOPreparation(const QUBOPreparation&) = delete;
    QUBOPreparation& operator=(const QUBOPreparation&) = delete;
    void set_memory_budget(std::size_t maximum_bytes);
    [[nodiscard]] IsingModel prepare(const BinaryQuadraticModel& bqm);
    [[nodiscard]] IsingModel prepare(
        std::size_t variables, const float* linear, std::size_t edges,
        const std::uint32_t* heads, const std::uint32_t* tails,
        const float* biases, float offset);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] IsingModel to_ising(const BinaryQuadraticModel& bqm);
[[nodiscard]] BinaryQuadraticModel load_qubo(const std::string& path);
void save_qubo(const BinaryQuadraticModel& bqm, const std::string& path);

}  // namespace sbm
