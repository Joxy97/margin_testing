#include "sbm/maxcut.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Size {
    std::size_t vertices;
    std::size_t edges;
};

int main(int argc, char** argv) {
    try {
        const fs::path output_dir = argc > 1 ? argv[1] : "benchmarks/maxcut/instances";
        const std::size_t maximum_vertices = argc > 2 ? std::stoull(argv[2]) : 2'000'000;
        const std::uint64_t base_seed = argc > 3 ? std::stoull(argv[3]) : 20260827;
        const std::vector<Size> sizes{
            {20, 160},          {64, 512},          {256, 2'048},
            {1'024, 8'192},     {10'000, 80'000},   {100'000, 800'000},
            {1'000'000, 8'000'000}, {2'000'000, 16'000'000},
        };

        fs::create_directories(output_dir);
        std::ofstream manifest(output_dir / "manifest.csv");
        if (!manifest) throw std::runtime_error("cannot write MaxCut manifest");
        manifest << "file,vertices,edges,average_degree,seed\n";

        for (std::size_t index = 0; index < sizes.size(); ++index) {
            const auto [vertices, edges] = sizes[index];
            if (vertices > maximum_vertices) continue;
            const int instances = vertices <= 100'000 ? 3 : 1;
            for (int replicate = 0; replicate < instances; ++replicate) {
                const auto seed = base_seed + index + 1'000ULL * replicate;
                std::ostringstream filename;
                filename << "maxcut_n" << std::setw(8) << std::setfill('0') << vertices
                         << "_m" << std::setw(9) << edges << "_seed" << seed << ".csv";
                const auto path = output_dir / filename.str();
                std::cerr << "generating " << filename.str() << "\n";
                auto graph = sbm::maxcut::generate(vertices, edges, seed);
                sbm::maxcut::save_csv(graph, path.string(), seed);
                manifest << filename.str() << ',' << vertices << ',' << graph.edges.size()
                         << ',' << (2.0 * graph.edges.size() / vertices) << ',' << seed << '\n';
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
