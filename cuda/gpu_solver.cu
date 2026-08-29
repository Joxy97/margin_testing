#include "sbm/solver.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace sbm {
namespace {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count != 0) {
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "cudaMalloc");
        }
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

private:
    T* data_ = nullptr;
    std::size_t count_;
};

__global__ void interaction_kernel(
    int n, const std::uint32_t* row_offsets, const std::uint32_t* columns,
    const float* couplings, const float* fields, const float* x, float* accum) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    float value = fields[row];
    for (std::uint32_t p = row_offsets[row]; p < row_offsets[row + 1]; ++p) {
        value += x[columns[p]] >= 0.0f ? couplings[p] : -couplings[p];
    }
    accum[row] = value;
}

__global__ void time_evolution_kernel(
    int n, float a, float a0, float dt, float c0, float gamma,
    const float* accum, float* x, float* y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float previous_y = y[i];
    float next_y = previous_y + ((a - a0) * x[i] + c0 * accum[i]) * dt;
    float next_x = x[i] + a0 * next_y * dt;
    if (fabsf(next_x) > 1.0f) {
        next_x = next_x >= 0.0f ? 1.0f : -1.0f;
        next_y = 0.0f;
    }
    y[i] = next_y + gamma * previous_y * dt;
    x[i] = next_x;
}

}  // namespace

SolverResult solve_gpu(const BinaryQuadraticModel& bqm, const SolverParameters& p) {
    if (bqm.size() == 0 || p.steps <= 0 || p.runs <= 0) {
        throw std::invalid_argument("BQM, steps, and runs must be non-empty/positive");
    }
    if (!(p.dt > 0.0) || !(p.a0 > 0.0) || p.gamma < 0.0) {
        throw std::invalid_argument("dt/a0 must be positive and gamma non-negative");
    }
    const auto model = to_ising(bqm);
    const auto n = model.size();
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        model.couplings.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("model is too large for the CUDA CSR index type");
    }

    std::vector<std::uint32_t> row_offsets(model.row_offsets.begin(), model.row_offsets.end());
    std::vector<float> couplings(model.couplings.begin(), model.couplings.end());
    std::vector<float> fields(model.fields.begin(), model.fields.end());
    DeviceBuffer<std::uint32_t> d_rows(row_offsets.size()), d_columns(model.columns.size());
    DeviceBuffer<float> d_couplings(couplings.size()), d_fields(fields.size());
    DeviceBuffer<float> d_x(n), d_y(n), d_accum(n);
    cuda_check(cudaMemcpy(d_rows.get(), row_offsets.data(), d_rows.bytes(), cudaMemcpyHostToDevice), "copy rows");
    if (d_columns.bytes() != 0) {
        cuda_check(cudaMemcpy(d_columns.get(), model.columns.data(), d_columns.bytes(), cudaMemcpyHostToDevice), "copy columns");
        cuda_check(cudaMemcpy(d_couplings.get(), couplings.data(), d_couplings.bytes(), cudaMemcpyHostToDevice), "copy couplings");
    }
    cuda_check(cudaMemcpy(d_fields.get(), fields.data(), d_fields.bytes(), cudaMemcpyHostToDevice), "copy fields");

    const float c0 = static_cast<float>(p.c0 > 0.0 ? p.c0 : estimate_c0(model));
    constexpr int threads = 256;
    const int blocks = (static_cast<int>(n) + threads - 1) / threads;
    SolverResult best;
    best.energy = std::numeric_limits<double>::infinity();

    for (int run = 0; run < p.runs; ++run) {
        std::mt19937_64 rng(
            p.seed + 0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(run));
        std::uniform_real_distribution<float> initial(
            -static_cast<float>(p.initial_scale), static_cast<float>(p.initial_scale));
        std::vector<float> x(n), y(n);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = initial(rng);
            y[i] = initial(rng);
        }
        cuda_check(cudaMemcpy(d_x.get(), x.data(), d_x.bytes(), cudaMemcpyHostToDevice), "copy x");
        cuda_check(cudaMemcpy(d_y.get(), y.data(), d_y.bytes(), cudaMemcpyHostToDevice), "copy y");

        const float da = static_cast<float>(p.a0 / p.steps);
        for (int step = 0; step < p.steps; ++step) {
            interaction_kernel<<<blocks, threads>>>(
                static_cast<int>(n), d_rows.get(), d_columns.get(), d_couplings.get(),
                d_fields.get(), d_x.get(), d_accum.get());
            time_evolution_kernel<<<blocks, threads>>>(
                static_cast<int>(n), step * da, static_cast<float>(p.a0),
                static_cast<float>(p.dt), c0, static_cast<float>(p.gamma),
                d_accum.get(), d_x.get(), d_y.get());
        }
        cuda_check(cudaGetLastError(), "launch dSB kernels");
        cuda_check(cudaMemcpy(x.data(), d_x.get(), d_x.bytes(), cudaMemcpyDeviceToHost), "copy result");

        std::vector<std::uint8_t> sample(n);
        for (std::size_t i = 0; i < n; ++i) sample[i] = x[i] >= 0.0f;
        const double energy = bqm.energy(sample);
        if (energy < best.energy) best = {std::move(sample), energy};
    }
    return best;
}

}  // namespace sbm
