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
CMAKE_SOURCE_DIR = /mnt/nvme0n1p1/honours-project/c-attempt

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/nvme0n1p1/honours-project/build

# Include any dependencies generated for this target.
include CMakeFiles/multicore_localization.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/multicore_localization.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/multicore_localization.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/multicore_localization.dir/flags.make

CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main.cpp.o: CMakeFiles/multicore_localization.dir/flags.make
CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main.cpp.o: /mnt/nvme0n1p1/honours-project/c-attempt/multicore-localization-inf-time/main.cpp
CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main.cpp.o: CMakeFiles/multicore_localization.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/nvme0n1p1/honours-project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main.cpp.o -MF CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main.cpp.o.d -o CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main.cpp.o -c /mnt/nvme0n1p1/honours-project/c-attempt/multicore-localization-inf-time/main.cpp

CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/nvme0n1p1/honours-project/c-attempt/multicore-localization-inf-time/main.cpp > CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main.cpp.i

CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/nvme0n1p1/honours-project/c-attempt/multicore-localization-inf-time/main.cpp -o CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main.cpp.s

CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_simple.cpp.o: CMakeFiles/multicore_localization.dir/flags.make
CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_simple.cpp.o: /mnt/nvme0n1p1/honours-project/c-attempt/multicore-localization-inf-time/main_simple.cpp
CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_simple.cpp.o: CMakeFiles/multicore_localization.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/nvme0n1p1/honours-project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_simple.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_simple.cpp.o -MF CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_simple.cpp.o.d -o CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_simple.cpp.o -c /mnt/nvme0n1p1/honours-project/c-attempt/multicore-localization-inf-time/main_simple.cpp

CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_simple.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_simple.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/nvme0n1p1/honours-project/c-attempt/multicore-localization-inf-time/main_simple.cpp > CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_simple.cpp.i

CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_simple.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_simple.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/nvme0n1p1/honours-project/c-attempt/multicore-localization-inf-time/main_simple.cpp -o CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_simple.cpp.s

CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_noise_response.cpp.o: CMakeFiles/multicore_localization.dir/flags.make
CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_noise_response.cpp.o: /mnt/nvme0n1p1/honours-project/c-attempt/multicore-localization-inf-time/main_noise_response.cpp
CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_noise_response.cpp.o: CMakeFiles/multicore_localization.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/nvme0n1p1/honours-project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_noise_response.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_noise_response.cpp.o -MF CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_noise_response.cpp.o.d -o CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_noise_response.cpp.o -c /mnt/nvme0n1p1/honours-project/c-attempt/multicore-localization-inf-time/main_noise_response.cpp

CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_noise_response.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_noise_response.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/nvme0n1p1/honours-project/c-attempt/multicore-localization-inf-time/main_noise_response.cpp > CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_noise_response.cpp.i

CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_noise_response.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_noise_response.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/nvme0n1p1/honours-project/c-attempt/multicore-localization-inf-time/main_noise_response.cpp -o CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_noise_response.cpp.s

# Object files for target multicore_localization
multicore_localization_OBJECTS = \
"CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main.cpp.o" \
"CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_simple.cpp.o" \
"CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_noise_response.cpp.o"

# External object files for target multicore_localization
multicore_localization_EXTERNAL_OBJECTS =

/mnt/nvme0n1p1/honours-project/c-attempt/multicore_localization: CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main.cpp.o
/mnt/nvme0n1p1/honours-project/c-attempt/multicore_localization: CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_simple.cpp.o
/mnt/nvme0n1p1/honours-project/c-attempt/multicore_localization: CMakeFiles/multicore_localization.dir/multicore-localization-inf-time/main_noise_response.cpp.o
/mnt/nvme0n1p1/honours-project/c-attempt/multicore_localization: CMakeFiles/multicore_localization.dir/build.make
/mnt/nvme0n1p1/honours-project/c-attempt/multicore_localization: /usr/lib64/libomp.so
/mnt/nvme0n1p1/honours-project/c-attempt/multicore_localization: /usr/lib64/libpthread.so
/mnt/nvme0n1p1/honours-project/c-attempt/multicore_localization: CMakeFiles/multicore_localization.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/nvme0n1p1/honours-project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable /mnt/nvme0n1p1/honours-project/c-attempt/multicore_localization"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/multicore_localization.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/multicore_localization.dir/build: /mnt/nvme0n1p1/honours-project/c-attempt/multicore_localization
.PHONY : CMakeFiles/multicore_localization.dir/build

CMakeFiles/multicore_localization.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/multicore_localization.dir/cmake_clean.cmake
.PHONY : CMakeFiles/multicore_localization.dir/clean

CMakeFiles/multicore_localization.dir/depend:
	cd /mnt/nvme0n1p1/honours-project/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/nvme0n1p1/honours-project/c-attempt /mnt/nvme0n1p1/honours-project/c-attempt /mnt/nvme0n1p1/honours-project/build /mnt/nvme0n1p1/honours-project/build /mnt/nvme0n1p1/honours-project/build/CMakeFiles/multicore_localization.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/multicore_localization.dir/depend

