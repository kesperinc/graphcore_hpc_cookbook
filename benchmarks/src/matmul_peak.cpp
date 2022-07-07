// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <cstdlib>

#include <iomanip>
#include <fstream>

#include <boost/program_options.hpp>
#include <cassert>
#include <cmath>
#include <string>

#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Program.hpp>
#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <poplin/MatMul.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>

#include <boost/timer/timer.hpp>

using ::std::map;
using ::std::vector;
using ::std::string;
using ::std::optional;

using ::poplar::BOOL;
using ::poplar::HALF;
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

    poplin::addCodelets(graph);
    popops::addCodelets(graph);
    return graph;
}

auto captureProfileInfo(Engine &engine) {
    std::ofstream graphOfs;
    graphOfs.open("graph.json", std::ofstream::out | std::ofstream::trunc);

    std::ofstream executionOfs;
    executionOfs.open("execution.json", std::ofstream::out | std::ofstream::trunc);
}

struct options
{
    int size;
    int iterations;
    int groups;
    int num_ipus;
    bool fr;
    bool ld128;
    bool msr;
    bool remap_out;
    std::string partials_type;
    std::string memory_proportion;
    poplar::Type type;

    options() : size(0), num_ipus(1)
    {}
};

static options parse_options(int argc, char* argv[], const char *desc)
{
    using boost::program_options::options_description;
    using boost::program_options::value;
    using boost::program_options::bool_switch;
    using boost::program_options::variables_map;
    using boost::program_options::validation_error;

    struct options ret;
    std::string type;

    options_description options(desc);
    options.add_options()
        ("help", "help message")
        ("size,s", value<int>(&ret.size)->default_value(1024), "size of matrix")
        ("iterations,i", value<int>(&ret.iterations)->default_value(10), "iterations")
        ("groups,g", value<int>(&ret.groups)->default_value(1), "iterations")
        ("type,t", value<std::string>(&type)->default_value("float"), "type")
        ("partials_type", value<std::string>(&ret.partials_type)->default_value("float"), "partial sum type")
        ("memory_proportion", value<std::string>(&ret.memory_proportion)->default_value("0.6"), "available memory proportion")
        ("fast_reduce", bool_switch(&ret.fr)->default_value(false), "Enable fast reduce")
        ("multi_stage_reduce", bool_switch(&ret.msr)->default_value(false), "Enable Multi stage reduce")
        ("remap_output_tensor", bool_switch(&ret.remap_out)->default_value(false), "Remap output tensor")
        ("128bit_load", bool_switch(&ret.ld128)->default_value(false), "128-bit load");
    variables_map vm;

    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, options), vm);
    notify(vm);
    if (vm.count("help")) {
        std::cout << options << std::endl;
        exit(EXIT_SUCCESS);
    }

    if (type == "float")
        ret.type = FLOAT;
    else if (type == "half")
        ret.type = HALF;
    else
        throw validation_error{validation_error::invalid_option_value, type, "type"};

    return ret;
}

int main(int argc, char *argv[]) {

    using boost::timer::cpu_timer;

    options opt = parse_options(argc, argv, "Peak MatMul FLOPS benchmark");
    const unsigned n = opt.size;
    const unsigned g = opt.groups;
    const auto type = opt.type;
    int num_ipus = opt.num_ipus;
    std::stringstream ss;

    std::cout << "Matrix " << n << "x" << n << " multiplication iterations = "
              << opt.iterations << std::endl;
    cpu_timer timer;
    auto device = getIpuDevice(num_ipus);
    if (!device.has_value()) {
        std::cerr << "Could not attach to an IPU device. Aborting" << std::endl;
        return EXIT_FAILURE;
    }
    timer.stop();
    ss << "Device attach: " << timer.format();
    auto graph = poplar::Graph(device->getTarget());

    poplin::addCodelets(graph);
    popops::addCodelets(graph);
    poprand::addCodelets(graph);
 
    Sequence seq, init;
    OptionFlags matmul_opt;
    poplin::matmul::PlanningCache cache;

    matmul_opt.set("enableMultiStageReduce", opt.msr ? "true" : "false");
    matmul_opt.set("enableFastReduce", opt.fr ? "true" : "false");
    matmul_opt.set("use128BitConvUnitLoad", opt.ld128 ? "true" : "false");
    matmul_opt.set("remapOutputTensor", opt.remap_out ? "true" : "false");
    matmul_opt.set("partialsType", opt.partials_type);
    matmul_opt.set("availableMemoryProportion", opt.memory_proportion);

    auto A = poplin::createMatMulGroupedInputLHS(graph, type, type, {g, n, n}, {g, n, n}, "A", matmul_opt, &cache);
    auto B = poplin::createMatMulGroupedInputRHS(graph, type, type, {g, n, n}, {g, n, n}, "B", matmul_opt, &cache);

    A = poprand::uniform(graph, NULL, 0, A, type, 0., 1., init, "randA");
    B = poprand::uniform(graph, NULL, 0, B, type, 0., 1., init, "randB");

    auto C = poplin::matMulGrouped(graph, A, B, seq, type, "C", matmul_opt, &cache);
    seq.add(Copy(C, B));
    timer.stop();
    ss << "Build program: " << timer.format();

    auto ENGINE_OPTIONS = OptionFlags{};
    timer.start();
    std::vector<Program> programs = {init, Repeat(opt.iterations, seq)};
    auto engine = Engine(graph, programs, ENGINE_OPTIONS);

    engine.load(*device);
    engine.enableExecutionProfiling();

    timer.stop();
    ss << "Graph compilation: " << timer.format();

    engine.run(0, "init");
    timer.start();
    engine.run(1, "compute");
    timer.stop();
    ss << opt.iterations << " Program runs took : " << timer.format();

    std::cout << ss.str();

    //2 * numRows **2 * numCols - numRows**2
    double flops = g*(2.f * n * n * n - n * n);
    auto elapsedSeconds = timer.elapsed().wall;
    std::cout << "elapsed time: " << std::fixed << std::setprecision(6)
              << elapsedSeconds / 1000000 << " ms" << std::endl;
    double avgTime = elapsedSeconds / opt.iterations;
    std::cout << "average kernel run time: " << std::fixed
              << std::setprecision(1) << avgTime / 1000 / g << " us"
              << std::endl;
    std::cout << "GFLOPS: " << std::fixed << std::setprecision(6)
              << flops / avgTime << std::endl;

    return EXIT_SUCCESS;
}

