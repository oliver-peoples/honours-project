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
include CMakeFiles/tem_localization_2d.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/tem_localization_2d.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/tem_localization_2d.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tem_localization_2d.dir/flags.make

CMakeFiles/tem_localization_2d.dir/tem_localization_2d.cpp.o: CMakeFiles/tem_localization_2d.dir/flags.make
CMakeFiles/tem_localization_2d.dir/tem_localization_2d.cpp.o: ../tem_localization_2d.cpp
CMakeFiles/tem_localization_2d.dir/tem_localization_2d.cpp.o: CMakeFiles/tem_localization_2d.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/finn/Documents/honours-project/c-attempt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tem_localization_2d.dir/tem_localization_2d.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/tem_localization_2d.dir/tem_localization_2d.cpp.o -MF CMakeFiles/tem_localization_2d.dir/tem_localization_2d.cpp.o.d -o CMakeFiles/tem_localization_2d.dir/tem_localization_2d.cpp.o -c /home/finn/Documents/honours-project/c-attempt/tem_localization_2d.cpp

CMakeFiles/tem_localization_2d.dir/tem_localization_2d.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tem_localization_2d.dir/tem_localization_2d.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/finn/Documents/honours-project/c-attempt/tem_localization_2d.cpp > CMakeFiles/tem_localization_2d.dir/tem_localization_2d.cpp.i

CMakeFiles/tem_localization_2d.dir/tem_localization_2d.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tem_localization_2d.dir/tem_localization_2d.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/finn/Documents/honours-project/c-attempt/tem_localization_2d.cpp -o CMakeFiles/tem_localization_2d.dir/tem_localization_2d.cpp.s

# Object files for target tem_localization_2d
tem_localization_2d_OBJECTS = \
"CMakeFiles/tem_localization_2d.dir/tem_localization_2d.cpp.o"

# External object files for target tem_localization_2d
tem_localization_2d_EXTERNAL_OBJECTS =

../tem_localization_2d: CMakeFiles/tem_localization_2d.dir/tem_localization_2d.cpp.o
../tem_localization_2d: CMakeFiles/tem_localization_2d.dir/build.make
../tem_localization_2d: CMakeFiles/tem_localization_2d.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/finn/Documents/honours-project/c-attempt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../tem_localization_2d"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tem_localization_2d.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tem_localization_2d.dir/build: ../tem_localization_2d
.PHONY : CMakeFiles/tem_localization_2d.dir/build

CMakeFiles/tem_localization_2d.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tem_localization_2d.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tem_localization_2d.dir/clean

CMakeFiles/tem_localization_2d.dir/depend:
	cd /home/finn/Documents/honours-project/c-attempt/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/finn/Documents/honours-project/c-attempt /home/finn/Documents/honours-project/c-attempt /home/finn/Documents/honours-project/c-attempt/build /home/finn/Documents/honours-project/c-attempt/build /home/finn/Documents/honours-project/c-attempt/build/CMakeFiles/tem_localization_2d.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tem_localization_2d.dir/depend

