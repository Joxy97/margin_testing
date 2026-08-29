#include "sbm/model.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace sbm {

double BinaryQuadraticModel::energy(const std::vector<std::uint8_t>& sample) const {
    if (sample.size() != size()) {
        throw std::invalid_argument("sample size does not match BQM");
    }
    double value = offset;
    for (std::size_t i = 0; i < size(); ++i) {
        value += linear[i] * static_cast<double>(sample[i] != 0);
    }
    for (const auto& term : quadratic) {
        value += term.bias * static_cast<double>(sample[term.u] != 0) *
                 static_cast<double>(sample[term.v] != 0);
    }
    return value;
}

double IsingModel::energy(const std::vector<std::int8_t>& spins) const {
    if (spins.size() != size()) {
        throw std::invalid_argument("spin size does not match Ising model");
    }
    double value = offset;
    for (std::size_t i = 0; i < size(); ++i) {
        value -= fields[i] * spins[i];
        for (std::size_t p = topology->row_offsets[i];
             p < topology->row_offsets[i + 1]; ++p) {
            value -= 0.5 * couplings[p] * spins[i] * spins[topology->columns[p]];
        }
    }
    return value;
}

IsingModel to_ising(const BinaryQuadraticModel& bqm) {
    const auto n = bqm.size();
    IsingModel model;
    auto topology = std::make_shared<IsingTopology>();
    model.topology = topology;
    model.fields.resize(n);
    model.offset = static_cast<float>(bqm.offset);

    std::vector<std::size_t> degrees(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        model.offset += 0.5F * static_cast<float>(bqm.linear[i]);
        model.fields[i] = -0.5F * static_cast<float>(bqm.linear[i]);
    }
    for (const auto& term : bqm.quadratic) {
        if (term.u >= n || term.v >= n || term.u == term.v) {
            throw std::invalid_argument("invalid off-diagonal quadratic bias");
        }
        const float bias = static_cast<float>(term.bias);
        const float j = -0.25F * bias;
        model.offset += 0.25F * bias;
        model.fields[term.u] -= 0.25F * bias;
        model.fields[term.v] -= 0.25F * bias;
        if (j != 0.0F) {
            ++degrees[term.u];
            ++degrees[term.v];
        }
    }

    topology->row_offsets.resize(n + 1);
    for (std::size_t i = 0; i < n; ++i) {
        topology->row_offsets[i + 1] = topology->row_offsets[i] + degrees[i];
    }
    topology->columns.resize(topology->row_offsets.back());
    model.couplings.resize(topology->row_offsets.back());
    auto cursor = topology->row_offsets;
    for (const auto& term : bqm.quadratic) {
        const float j = -0.25F * static_cast<float>(term.bias);
        if (j == 0.0F) continue;
        const auto uv = cursor[term.u]++;
        const auto vu = cursor[term.v]++;
        topology->columns[uv] = term.v;
        model.couplings[uv] = j;
        topology->columns[vu] = term.u;
        model.couplings[vu] = j;
    }
    return model;
}

BinaryQuadraticModel load_qubo(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open QUBO file: " + path);
    }

    BinaryQuadraticModel bqm;
    std::map<std::pair<std::uint32_t, std::uint32_t>, double> terms;
    std::string tag;
    while (input >> tag) {
        if (tag[0] == '#') {
            input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        } else if (tag == "p") {
            std::string kind;
            std::size_t n;
            input >> kind >> n;
            if (kind != "qubo") {
                throw std::runtime_error("expected 'p qubo <variables>'");
            }
            bqm.linear.assign(n, 0.0);
        } else if (tag == "o") {
            input >> bqm.offset;
        } else if (tag == "l") {
            std::size_t i;
            double bias;
            input >> i >> bias;
            if (i >= bqm.size()) {
                throw std::runtime_error("linear-bias index is out of range");
            }
            bqm.linear[i] += bias;
        } else if (tag == "q") {
            std::uint32_t u, v;
            double bias;
            input >> u >> v >> bias;
            if (u >= bqm.size() || v >= bqm.size()) {
                throw std::runtime_error("quadratic-bias index is out of range");
            }
            if (u == v) {
                bqm.linear[u] += bias;
            } else {
                if (u > v) std::swap(u, v);
                terms[{u, v}] += bias;
            }
        } else {
            throw std::runtime_error("unknown QUBO record: " + tag);
        }
        if (!input) {
            throw std::runtime_error("malformed QUBO file: " + path);
        }
    }
    for (const auto& [indices, bias] : terms) {
        if (bias != 0.0) {
            bqm.quadratic.push_back({indices.first, indices.second, bias});
        }
    }
    if (bqm.linear.empty()) {
        throw std::runtime_error("QUBO file has no problem header");
    }
    return bqm;
}

void save_qubo(const BinaryQuadraticModel& bqm, const std::string& path) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write QUBO file: " + path);
    }
    output << std::setprecision(17);
    output << "p qubo " << bqm.size() << '\n';
    output << "o " << bqm.offset << '\n';
    for (std::size_t i = 0; i < bqm.size(); ++i) {
        if (bqm.linear[i] != 0.0) output << "l " << i << ' ' << bqm.linear[i] << '\n';
    }
    for (const auto& term : bqm.quadratic) {
        if (term.bias != 0.0) {
            output << "q " << term.u << ' ' << term.v << ' ' << term.bias << '\n';
        }
    }
}

}  // namespace sbm
