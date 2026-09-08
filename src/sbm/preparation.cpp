#include "sbm/model.hpp"
#include <algorithm>
#include <cmath>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {
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
            if (candidate->hash == hash && candidate->topology->row_offsets.size() == variables + 1 && candidate->heads.size() == edges &&
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

namespace sbm {
struct QUBOPreparation::Impl {
    TopologyCache cache;
    std::size_t maximum_bytes = 0;
};

QUBOPreparation::QUBOPreparation(std::size_t maximum_bytes) : impl_(std::make_unique<Impl>()) {
    impl_->maximum_bytes = maximum_bytes;
}
QUBOPreparation::~QUBOPreparation() = default;
void QUBOPreparation::set_memory_budget(std::size_t maximum_bytes) { impl_->maximum_bytes = maximum_bytes; }

IsingModel QUBOPreparation::prepare(
    std::size_t variables, const float* linear, std::size_t edges,
    const std::uint32_t* heads, const std::uint32_t* tails, const float* biases, float offset) {
    if ((variables && !linear) || (edges && (!heads || !tails || !biases)))
        throw std::invalid_argument("null QUBO coefficients");
    if (!std::isfinite(offset)) throw std::invalid_argument("nonfinite QUBO offset");
    for (std::size_t i = 0; i < variables; ++i)
        if (!std::isfinite(linear[i])) throw std::invalid_argument("nonfinite QUBO linear bias");
    for (std::size_t i = 0; i < edges; ++i)
        if (!std::isfinite(biases[i])) throw std::invalid_argument("nonfinite QUBO quadratic bias");
    const auto topology = impl_->cache.get(variables, heads, tails, edges, impl_->maximum_bytes);
    return build_ising_model(topology, linear, biases, offset);
}

IsingModel QUBOPreparation::prepare(const BinaryQuadraticModel& bqm) {
    std::vector<float> linear(bqm.linear.begin(), bqm.linear.end());
    std::vector<std::uint32_t> heads, tails;
    std::vector<float> biases;
    heads.reserve(bqm.quadratic.size()); tails.reserve(bqm.quadratic.size()); biases.reserve(bqm.quadratic.size());
    for (const auto& term : bqm.quadratic) {
        heads.push_back(term.u); tails.push_back(term.v); biases.push_back(static_cast<float>(term.bias));
    }
    return prepare(bqm.size(), linear.data(), biases.size(), heads.data(), tails.data(), biases.data(), static_cast<float>(bqm.offset));
}
}  // namespace sbm
