// Thin driver for the pinned, unmodified authors' exact MCQDW implementation.
// Build with: g++ -O3 -DNDEBUG -std=c++23 -I AUTHOR_SOURCE/lib this_file.cpp -o verify_author
// NDEBUG matches the authors' CMake, including its Debug configuration.
// No time/step limit and no benchmark solver is invoked here.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "glib/algorithms/mcqd.hpp"

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Expected one weighted DIMACS input path");
        std::ifstream file(argv[1]);
        if (!file) throw std::runtime_error("Cannot read public input");
        insilab::helper::Array2d<bool> adjacency;
        std::vector<insilab::glib::weight_t> weights;
        std::vector<bool> seen_weights;
        std::size_t declared_edges = 0, actual_edges = 0, actual_weights = 0;
        long long total_weight = 0;
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream record(line);
            char type;
            if (!(record >> type) || type == 'c' || type == '#') continue;
            if (type == 'p') {
                std::string kind;
                std::size_t n;
                if (!adjacency.empty() || !(record >> kind >> n >> declared_edges) || kind != "edge" || n == 0)
                    throw std::runtime_error("Invalid or duplicate DIMACS header");
                adjacency = insilab::helper::Array2d<bool>(n);
                weights.resize(n);
                seen_weights.resize(n, false);
            } else if (type == 'n') {
                std::size_t label;
                long long weight;
                if (!(record >> label >> weight) || label < 1 || label > weights.size() || weight <= 0 || seen_weights[label-1])
                    throw std::runtime_error("Invalid public vertex weight");
                if (weight > (1LL << 53) || total_weight > (1LL << 53) - weight)
                    throw std::runtime_error("Integer bounds are not exact in authors' binary64 weight arithmetic");
                total_weight += weight;
                weights[label-1] = weight;
                seen_weights[label-1] = true;
                ++actual_weights;
            } else if (type == 'e') {
                std::size_t i, j;
                if (!(record >> i >> j) || i < 1 || j < 1 || i > weights.size() || j > weights.size() || i == j)
                    throw std::runtime_error("Invalid public graph edge");
                if (adjacency.get(i-1, j-1)) throw std::runtime_error("Duplicate undirected edge");
                adjacency.set(i-1, j-1);
                adjacency.set(j-1, i-1);
                ++actual_edges;
            } else throw std::runtime_error("Unknown DIMACS record");
            std::string extra;
            if (record >> extra) throw std::runtime_error("Trailing DIMACS fields");
        }
        if (adjacency.empty() || actual_edges != declared_edges || actual_weights != weights.size())
            throw std::runtime_error("Public DIMACS counts do not match header");
        const auto start = std::chrono::steady_clock::now();
        // Default max_steps is std::nullopt: run the complete exact search.
        auto [clique, reported] = insilab::glib::algorithms::find_maximum_weight_clique(adjacency, weights);
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        long long independently_summed = 0;
        for (auto vertex : clique) independently_summed += static_cast<long long>(weights[vertex]);
        if (reported != static_cast<double>(independently_summed))
            throw std::runtime_error("Authors' reported weight differs from independent integer sum");
        for (std::size_t i = 0; i < clique.size(); ++i)
            for (std::size_t j = i+1; j < clique.size(); ++j)
                if (!adjacency.get(clique[i], clique[j])) throw std::runtime_error("Result is not a clique");
        std::sort(clique.begin(), clique.end());
        std::cout << std::setprecision(17)
                  << "{\"completed\":true,\"exact\":true,\"method\":\"MCQDW\",\"configured_step_limit\":null,"
                  << "\"configured_time_limit\":null,\"weight_integer\":" << independently_summed
                  << ",\"elapsed_search_s\":" << elapsed
                  << ",\"binary64_integer_bound\":" << total_weight
                  << ",\"source_vertex_labels\":[";
        for (std::size_t i = 0; i < clique.size(); ++i)
            std::cout << (i ? "," : "") << clique[i]+1;
        std::cout << "]}" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 2;
    }
}
