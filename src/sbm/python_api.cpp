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

struct CachedTopology {
    std::size_t hash = 0;
    std::vector<std::uint32_t> heads;
    std::vector<std::uint32_t> tails;
    std::shared_ptr<const sbm::IsingTopology> topology;

    [[nodiscard]] std::size_t bytes() const noexcept {
        return heads.size() * sizeof(std::uint32_t) +
               tails.size() * sizeof(std::uint32_t) +
               topology->row_offsets.size() * sizeof(std::size_t) +
               topology->columns.size() * sizeof(std::uint32_t);
    }
};

std::size_t topology_hash(
    std::size_t variables, const std::uint32_t* heads,
    const std::uint32_t* tails, std::size_t edges) {
    std::size_t hash = variables ^ (edges + 0x9e3779b97f4a7c15ULL);
    for (std::size_t edge = 0; edge < edges; ++edge) {
        hash ^= static_cast<std::size_t>(heads[edge]) + 0x9e3779b97f4a7c15ULL +
                (hash << 6U) + (hash >> 2U);
        hash ^= static_cast<std::size_t>(tails[edge]) + 0x9e3779b97f4a7c15ULL +
                (hash << 6U) + (hash >> 2U);
    }
    return hash;
}

std::shared_ptr<CachedTopology> build_topology(
    std::size_t variables, const std::uint32_t* heads,
    const std::uint32_t* tails, std::size_t edges, std::size_t hash) {
    auto cached = std::make_shared<CachedTopology>();
    cached->hash = hash;
    if (edges != 0) {
        cached->heads.assign(heads, heads + edges);
        cached->tails.assign(tails, tails + edges);
    }
    auto topology = std::make_shared<sbm::IsingTopology>();
    std::vector<std::size_t> degrees(variables, 0);
    for (std::size_t edge = 0; edge < edges; ++edge) {
        const auto u = heads[edge];
        const auto v = tails[edge];
        if (u >= variables || v >= variables || u == v) {
            throw std::invalid_argument("invalid off-diagonal quadratic topology");
        }
        ++degrees[u];
        ++degrees[v];
    }
    topology->row_offsets.resize(variables + 1);
    for (std::size_t variable = 0; variable < variables; ++variable) {
        topology->row_offsets[variable + 1] =
            topology->row_offsets[variable] + degrees[variable];
    }
    topology->columns.resize(topology->row_offsets.back());
    auto cursor = topology->row_offsets;
    for (std::size_t edge = 0; edge < edges; ++edge) {
        const auto u = heads[edge];
        const auto v = tails[edge];
        topology->columns[cursor[u]++] = v;
        topology->columns[cursor[v]++] = u;
    }
    cached->topology = std::move(topology);
    return cached;
}

class TopologyCache {
public:
    std::shared_ptr<CachedTopology> get(
        std::size_t variables, const std::uint32_t* heads,
        const std::uint32_t* tails, std::size_t edges,
        std::size_t maximum_bytes) {
        const auto hash = topology_hash(variables, heads, tails, edges);
        std::lock_guard<std::mutex> lock(mutex_);
        maximum_bytes_ = maximum_bytes;
        evict();
        for (auto iterator = entries_.begin(); iterator != entries_.end(); ++iterator) {
            const auto& candidate = *iterator;
            if (candidate->hash == hash && candidate->heads.size() == edges &&
                std::equal(candidate->heads.begin(), candidate->heads.end(), heads) &&
                std::equal(candidate->tails.begin(), candidate->tails.end(), tails)) {
                entries_.splice(entries_.begin(), entries_, iterator);
                return candidate;
            }
        }
        auto candidate = build_topology(variables, heads, tails, edges, hash);
        if (maximum_bytes_ != 0 && candidate->bytes() <= maximum_bytes_) {
            current_bytes_ += candidate->bytes();
            entries_.push_front(candidate);
            evict();
        }
        return candidate;
    }

private:
    void evict() {
        while (!entries_.empty() &&
               (maximum_bytes_ == 0 || current_bytes_ > maximum_bytes_)) {
            current_bytes_ -= entries_.back()->bytes();
            entries_.pop_back();
        }
    }

    std::mutex mutex_;
    std::list<std::shared_ptr<CachedTopology>> entries_;
    std::size_t current_bytes_ = 0;
    std::size_t maximum_bytes_ = 0;
};

TopologyCache topology_cache;

sbm::IsingModel build_ising_model(
    const std::shared_ptr<CachedTopology>& cached, const float* linear,
    const float* biases, float offset) {
    sbm::IsingModel model;
    model.topology = cached->topology;
    model.fields.resize(cached->topology->row_offsets.size() - 1);
    model.couplings.resize(cached->topology->columns.size());
    model.offset = offset;
    for (std::size_t variable = 0; variable < model.size(); ++variable) {
        model.offset += 0.5F * linear[variable];
        model.fields[variable] = -0.5F * linear[variable];
    }
    auto cursor = cached->topology->row_offsets;
    for (std::size_t edge = 0; edge < cached->heads.size(); ++edge) {
        const auto u = cached->heads[edge];
        const auto v = cached->tails[edge];
        const float bias = biases[edge];
        const float coupling = -0.25F * bias;
        model.offset += 0.25F * bias;
        model.fields[u] -= 0.25F * bias;
        model.fields[v] -= 0.25F * bias;
        model.couplings[cursor[u]++] = coupling;
        model.couplings[cursor[v]++] = coupling;
    }
    return model;
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

        std::vector<sbm::IsingModel> models;
        std::vector<std::vector<std::uint8_t>> warm_starts;
        models.reserve(problem_count);
        if (initial_sample_flags != nullptr) warm_starts.resize(problem_count);
        for (std::size_t problem = 0; problem < problem_count; ++problem) {
            const auto variable_start = variable_offsets[problem];
            const auto variable_stop = variable_offsets[problem + 1];
            const auto edge_start = quadratic_offsets[problem];
            const auto edge_stop = quadratic_offsets[problem + 1];
            auto topology = topology_cache.get(
                variable_stop - variable_start,
                quadratic_u + edge_start,
                quadratic_v + edge_start,
                edge_stop - edge_start,
                topology_cache_bytes);
            models.push_back(build_ising_model(
                topology,
                linear + variable_start,
                quadratic_bias + edge_start,
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

        const auto batches = sbm::solve_cpu_ising_candidates_batch(
            models, parameters, warm_starts);
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
