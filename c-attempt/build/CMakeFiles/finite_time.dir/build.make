# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/finn/Documents/honours-project/c-attempt

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/finn/Documents/honours-project/c-attempt/build

# Include any dependencies generated for this target.
include CMakeFiles/finite_time.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/finite_time.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/finite_time.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/finite_time.dir/flags.make

CMakeFiles/finite_time.dir/finite-time/main.cpp.o: CMakeFiles/finite_time.dir/flags.make
CMakeFiles/finite_time.dir/finite-time/main.cpp.o: ../finite-time/main.cpp
CMakeFiles/finite_time.dir/finite-time/main.cpp.o: CMakeFiles/finite_time.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/finn/Documents/honours-project/c-attempt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/finite_time.dir/finite-time/main.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/finite_time.dir/finite-time/main.cpp.o -MF CMakeFiles/finite_time.dir/finite-time/main.cpp.o.d -o CMakeFiles/finite_time.dir/finite-time/main.cpp.o -c /home/finn/Documents/honours-project/c-attempt/finite-time/main.cpp

CMakeFiles/finite_time.dir/finite-time/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/finite_time.dir/finite-time/main.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/finn/Documents/honours-project/c-attempt/finite-time/main.cpp > CMakeFiles/finite_time.dir/finite-time/main.cpp.i

CMakeFiles/finite_time.dir/finite-time/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/finite_time.dir/finite-time/main.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/finn/Documents/honours-project/c-attempt/finite-time/main.cpp -o CMakeFiles/finite_time.dir/finite-time/main.cpp.s

# Object files for target finite_time
finite_time_OBJECTS = \
"CMakeFiles/finite_time.dir/finite-time/main.cpp.o"

# External object files for target finite_time
finite_time_EXTERNAL_OBJECTS =

../finite_time: CMakeFiles/finite_time.dir/finite-time/main.cpp.o
../finite_time: CMakeFiles/finite_time.dir/build.make
../finite_time: /usr/lib64/libomp.so
../finite_time: /usr/lib64/libpthread.so
../finite_time: CMakeFiles/finite_time.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/finn/Documents/honours-project/c-attempt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../finite_time"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/finite_time.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/finite_time.dir/build: ../finite_time
.PHONY : CMakeFiles/finite_time.dir/build

CMakeFiles/finite_time.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/finite_time.dir/cmake_clean.cmake
.PHONY : CMakeFiles/finite_time.dir/clean

CMakeFiles/finite_time.dir/depend:
	cd /home/finn/Documents/honours-project/c-attempt/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/finn/Documents/honours-project/c-attempt /home/finn/Documents/honours-project/c-attempt /home/finn/Documents/honours-project/c-attempt/build /home/finn/Documents/honours-project/c-attempt/build /home/finn/Documents/honours-project/c-attempt/build/CMakeFiles/finite_time.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/finite_time.dir/depend

