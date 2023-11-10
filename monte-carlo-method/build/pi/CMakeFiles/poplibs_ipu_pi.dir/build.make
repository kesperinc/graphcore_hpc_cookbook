# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/sunkim317/.local/lib/python3.8/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/sunkim317/.local/lib/python3.8/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sunkim317/hpc-cookbook/monte-carlo-method

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sunkim317/hpc-cookbook/monte-carlo-method/build

# Include any dependencies generated for this target.
include pi/CMakeFiles/poplibs_ipu_pi.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include pi/CMakeFiles/poplibs_ipu_pi.dir/compiler_depend.make

# Include the progress variables for this target.
include pi/CMakeFiles/poplibs_ipu_pi.dir/progress.make

# Include the compile flags for this target's objects.
include pi/CMakeFiles/poplibs_ipu_pi.dir/flags.make

pi/CMakeFiles/poplibs_ipu_pi.dir/poplibs_ipu_pi.cpp.o: pi/CMakeFiles/poplibs_ipu_pi.dir/flags.make
pi/CMakeFiles/poplibs_ipu_pi.dir/poplibs_ipu_pi.cpp.o: /home/sunkim317/hpc-cookbook/monte-carlo-method/pi/poplibs_ipu_pi.cpp
pi/CMakeFiles/poplibs_ipu_pi.dir/poplibs_ipu_pi.cpp.o: pi/CMakeFiles/poplibs_ipu_pi.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sunkim317/hpc-cookbook/monte-carlo-method/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object pi/CMakeFiles/poplibs_ipu_pi.dir/poplibs_ipu_pi.cpp.o"
	cd /home/sunkim317/hpc-cookbook/monte-carlo-method/build/pi && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT pi/CMakeFiles/poplibs_ipu_pi.dir/poplibs_ipu_pi.cpp.o -MF CMakeFiles/poplibs_ipu_pi.dir/poplibs_ipu_pi.cpp.o.d -o CMakeFiles/poplibs_ipu_pi.dir/poplibs_ipu_pi.cpp.o -c /home/sunkim317/hpc-cookbook/monte-carlo-method/pi/poplibs_ipu_pi.cpp

pi/CMakeFiles/poplibs_ipu_pi.dir/poplibs_ipu_pi.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/poplibs_ipu_pi.dir/poplibs_ipu_pi.cpp.i"
	cd /home/sunkim317/hpc-cookbook/monte-carlo-method/build/pi && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sunkim317/hpc-cookbook/monte-carlo-method/pi/poplibs_ipu_pi.cpp > CMakeFiles/poplibs_ipu_pi.dir/poplibs_ipu_pi.cpp.i

pi/CMakeFiles/poplibs_ipu_pi.dir/poplibs_ipu_pi.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/poplibs_ipu_pi.dir/poplibs_ipu_pi.cpp.s"
	cd /home/sunkim317/hpc-cookbook/monte-carlo-method/build/pi && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sunkim317/hpc-cookbook/monte-carlo-method/pi/poplibs_ipu_pi.cpp -o CMakeFiles/poplibs_ipu_pi.dir/poplibs_ipu_pi.cpp.s

# Object files for target poplibs_ipu_pi
poplibs_ipu_pi_OBJECTS = \
"CMakeFiles/poplibs_ipu_pi.dir/poplibs_ipu_pi.cpp.o"

# External object files for target poplibs_ipu_pi
poplibs_ipu_pi_EXTERNAL_OBJECTS =

pi/poplibs_ipu_pi: pi/CMakeFiles/poplibs_ipu_pi.dir/poplibs_ipu_pi.cpp.o
pi/poplibs_ipu_pi: pi/CMakeFiles/poplibs_ipu_pi.dir/build.make
pi/poplibs_ipu_pi: /home/sunkim317/poplar_sdk-ubuntu_20_04-3.3.0+1403-208993bbb7/poplar-ubuntu_20_04-3.3.0+7857-b67b751185/lib/libpoplar.so
pi/poplibs_ipu_pi: pi/CMakeFiles/poplibs_ipu_pi.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/sunkim317/hpc-cookbook/monte-carlo-method/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable poplibs_ipu_pi"
	cd /home/sunkim317/hpc-cookbook/monte-carlo-method/build/pi && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/poplibs_ipu_pi.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
pi/CMakeFiles/poplibs_ipu_pi.dir/build: pi/poplibs_ipu_pi
.PHONY : pi/CMakeFiles/poplibs_ipu_pi.dir/build

pi/CMakeFiles/poplibs_ipu_pi.dir/clean:
	cd /home/sunkim317/hpc-cookbook/monte-carlo-method/build/pi && $(CMAKE_COMMAND) -P CMakeFiles/poplibs_ipu_pi.dir/cmake_clean.cmake
.PHONY : pi/CMakeFiles/poplibs_ipu_pi.dir/clean

pi/CMakeFiles/poplibs_ipu_pi.dir/depend:
	cd /home/sunkim317/hpc-cookbook/monte-carlo-method/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sunkim317/hpc-cookbook/monte-carlo-method /home/sunkim317/hpc-cookbook/monte-carlo-method/pi /home/sunkim317/hpc-cookbook/monte-carlo-method/build /home/sunkim317/hpc-cookbook/monte-carlo-method/build/pi /home/sunkim317/hpc-cookbook/monte-carlo-method/build/pi/CMakeFiles/poplibs_ipu_pi.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : pi/CMakeFiles/poplibs_ipu_pi.dir/depend

