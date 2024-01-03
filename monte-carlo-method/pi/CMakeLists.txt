macro(add_benchmark)
        set(singleValue NAME)
        set(multipleValues SOURCES)
        cmake_parse_arguments(local "" "${singleValue}" "${multipleValues}" ${ARGN})

        add_executable(${local_NAME} ${local_SOURCES})
	include_directories("/home/sunkim317/.local/include")
	link_directories("/home/sunkim317/.local/lib")
        target_link_libraries(${local_NAME} boost_program_options poplar poputil popops poprand )
endmacro()

add_benchmark(
        NAME vertex_ipu_pi
        SOURCES vertex_ipu_pi.cpp)
add_benchmark(
        NAME iterative_ipu_pi
        SOURCES iterative_ipu_pi.cpp)

