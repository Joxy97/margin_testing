#include "sbm/model.hpp"
#include "sbm/solver.hpp"

#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void usage(const char* program) {
    std::cerr
        << "Usage: " << program << " MODEL.qubo [options]\n"
        << "  --steps N --runs N --dt X --a0 X --c0 X --gamma X --seed N\n";
}

template <class T>
T number(const char* text);

template <>
int number<int>(const char* text) { return std::stoi(text); }
template <>
double number<double>(const char* text) { return std::stod(text); }
template <>
std::uint64_t number<std::uint64_t>(const char* text) { return std::stoull(text); }

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            usage(argv[0]);
            return 2;
        }
        const std::string path = argv[1];
        sbm::SolverParameters parameters;
        for (int i = 2; i < argc; ++i) {
            const std::string option = argv[i];
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + option);
            const char* value = argv[++i];
            if (option == "--steps") parameters.steps = number<int>(value);
            else if (option == "--runs") parameters.runs = number<int>(value);
            else if (option == "--dt") parameters.dt = number<double>(value);
            else if (option == "--a0") parameters.a0 = number<double>(value);
            else if (option == "--c0") parameters.c0 = number<double>(value);
            else if (option == "--gamma") parameters.gamma = number<double>(value);
            else if (option == "--seed") parameters.seed = number<std::uint64_t>(value);
            else throw std::invalid_argument("unknown option: " + option);
        }

        const auto bqm = sbm::load_qubo(path);
        const auto result = sbm::solve_cpu(bqm, parameters);

        std::cout << std::setprecision(17) << "energy " << result.energy << "\nsample";
        for (auto bit : result.sample) std::cout << ' ' << static_cast<int>(bit);
        std::cout << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
