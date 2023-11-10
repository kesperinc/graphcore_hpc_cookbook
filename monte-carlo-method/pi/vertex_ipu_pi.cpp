// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "pi_options.hpp"

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <map>
#include <random>
#include <chrono>

#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Program.hpp>

using ::std::map;
using ::std::vector;
using ::std::string;
using ::std::optional;

using ::poplar::BOOL;
using ::poplar::FLOAT;
using ::poplar::UNSIGNED_INT;
using ::poplar::OptionFlags;
using ::poplar::Tensor;
using ::poplar::Graph;
using ::poplar::Engine;
using ::poplar::Device;
using ::poplar::DeviceManager;
using ::poplar::TargetType;
using ::poplar::program::Program;
using ::poplar::program::Sequence;
using ::poplar::program::Copy;
using ::poplar::program::Repeat;
using ::poplar::program::Execute;

static const auto MAX_TENSOR_SIZE = 55000000ul;

auto getIpuDevice(const unsigned int numIpus = 1) -> optional<Device> {
    DeviceManager manager = DeviceManager::createDeviceManager();
    optional<Device> device = std::nullopt;
    for (auto &d : manager.getDevices(TargetType::IPU, numIpus)) {
        std::cout << "Trying to attach to IPU " << d.getId();
        if (d.attach()) {
            std::cout << " - attached" << std::endl;
            device = {std::move(d)};
            break;
        } else {
            std::cout << std::endl << "Error attaching to device" << std::endl;
        }
    }
    return device;
}

auto createGraphAndAddCodelets(const optional<Device> &device) -> Graph {
    auto graph = poplar::Graph(device->getTarget());

    graph.addCodelets({"pi_vertex.cpp"}, "-O3");
    return graph;
}

auto serializeGraph(const Graph &graph) {
    std::ofstream graphSerOfs;
    graphSerOfs.open("serialized_graph.capnp", std::ofstream::out | std::ofstream::trunc);

    graph.serialize(graphSerOfs, poplar::SerializationFormat::Binary);
}

auto captureProfileInfo(Engine &engine) {
    std::ofstream graphOfs;
    graphOfs.open("graph.json", std::ofstream::out | std::ofstream::trunc);

    std::ofstream executionOfs;
    executionOfs.open("execution.json", std::ofstream::out | std::ofstream::trunc);
}

int main(int argc, char *argv[]) {
    pi_options options = parse_options(argc, argv, "IPU PI Iterative");
    auto precision = options.precision;
    auto iterations = options.iterations;

    std::cout << "STEP 1: Connecting to an IPU device" << std::endl;
    auto device = getIpuDevice(options.num_ipus);
    if (!device.has_value()) {
        std::cerr << "Could not attach to an IPU device. Aborting" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "STEP 2: Create graph and compile codelets" << std::endl;
    auto graph = createGraphAndAddCodelets(device);

    std::cout << "STEP 3: Define data streams" << std::endl;
    size_t numTiles = device->getTarget().getNumTiles();
    auto fromIpuStream = graph.addDeviceToHostFIFO("FROM_IPU", UNSIGNED_INT, numTiles * 6); //Device to host FIFO 스트림을 만들어서 여러 ipu를 사용

    std::cout << "STEP 4: Building the compute graph" << std::endl;
    auto counts = graph.addVariable(UNSIGNED_INT, {numTiles * 6}, "counts");
    poputil::mapTensorLinearly(graph, counts);          // vertex를 직선으로 나열하여 정리 각 슬라이스에서 처리한 counts 

    const auto NumElemsPerTile = iterations / (numTiles * 6);
    auto cs = graph.addComputeSet("loopBody");          //loopBody computeset을 설정 
    std::cout << "numTiles = " << numTiles << std::endl;
    std::cout << "NumElemsPerTile = " << NumElemsPerTile << std::endl;

    for (auto tileNum = 0u; tileNum < numTiles; tileNum++) {
        const auto sliceStart = tileNum * 6;             //tile을 0~6, 7~12 등 6개 단위로 나눈다 
        const auto sliceEnd = (tileNum + 1) * 6; 

        auto v = graph.addVertex(cs, "PiVertex", {
                {"hits", counts.slice(sliceStart, sliceEnd)} //PiVertex computeset을 만들고 vertex를 슬라이스에 할당한다. 
        });
        graph.setInitialValue(v["iterations"], NumElemsPerTile);
        graph.setPerfEstimate(v, 10); // Ideally you'd get this as right as possible
        graph.setTileMapping(v, tileNum);
    }
 
    std::cout << "STEP 5: Create engine and compile graph" << std::endl;
    auto ENGINE_OPTIONS = OptionFlags{
            {"target.saveArchive",                "archive.a"},
            {"debug.instrument",                  "true"},
            {"debug.instrumentCompute",           "true"},
            {"debug.instrumentControlFlow",       "true"},
            {"debug.computeInstrumentationLevel", "tile"},
            {"debug.outputAllSymbols",            "true"},
            {"autoReport.all",                    "true"},
            {"autoReport.outputLoweredVars",      "true"},
            {"autoReport.outputSerializedGraph",  "true"},
            {"debug.retainDebugInformation",      "true"}
    };
    auto engine = Engine(graph, Sequence({Execute(cs), Copy(counts, fromIpuStream)}), ENGINE_OPTIONS);
        
    std::cout << "STEP 6: Load compiled graph onto the IPU tiles" << std::endl;
    engine.load(*device);
    engine.enableExecutionProfiling();

    std::cout << "STEP 7: Attach data streams" << std::endl;
    
    std::cout << "iterations (bytes)= " << iterations << std::endl;
    // auto results = std::vector<unsigned int>(numTiles * 6); 
    auto results = std::vector<unsigned int>(iterations); 
    engine.connectStream("FROM_IPU", results.data(), results.data() + results.size());  // IPU에서 받아 results에 저장
    std::cout << "results.size = " << results.size() << std::endl;
    std::cout << "results = " << results[0] << std::endl;

    std::cout << "STEP 8: Run programs" << std::endl;
    auto hits = 0ull;  //unsigned long long
    
    auto start = std::chrono::steady_clock::now();
    engine.run(0, "main"); // Main program

    auto stop = std::chrono::steady_clock::now();
    for (size_t i = 0; i < results.size(); i++) {
        // hits += results[i];
        std::cout << "rand = " << results[i] << " " << (float) results[i]/(float)UINT32_MAX << std::endl;
    }

    std::cout << "STEP 9: Capture debug and profile info" << std::endl;
    serializeGraph(graph);
    captureProfileInfo(engine);
    engine.printProfileSummary(std::cout,
                               OptionFlags{{"showExecutionSteps", "false"}});
    std::cout << std::endl;
    std::cout << *std::max_element(results.begin(), results.end()) << std::endl;
    std::cout << "chunk_size = " << numTiles * 6 << " repeats = " << iterations / numTiles * 6 << std::endl; 
    std::cout << "tests = " << iterations << " took " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " us" << std::endl; 
    std::cout << "pi = " << std::setprecision(precision) << (4. * hits/(iterations)) << std::endl;

    return EXIT_SUCCESS;
}
