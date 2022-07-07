add_executable(matmul_peak matmul_peak.cpp)

target_link_libraries(matmul_peak
        poplar
        poputil
        popops
        poplin
        poprand
        boost_program_options
        boost_timer
        )
