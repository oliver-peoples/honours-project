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
include CMakeFiles/xy_form_tem.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/xy_form_tem.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/xy_form_tem.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/xy_form_tem.dir/flags.make

CMakeFiles/xy_form_tem.dir/xy-form-tem/main.cpp.o: CMakeFiles/xy_form_tem.dir/flags.make
CMakeFiles/xy_form_tem.dir/xy-form-tem/main.cpp.o: /mnt/nvme0n1p1/honours-project/c-attempt/xy-form-tem/main.cpp
CMakeFiles/xy_form_tem.dir/xy-form-tem/main.cpp.o: CMakeFiles/xy_form_tem.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/nvme0n1p1/honours-project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/xy_form_tem.dir/xy-form-tem/main.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/xy_form_tem.dir/xy-form-tem/main.cpp.o -MF CMakeFiles/xy_form_tem.dir/xy-form-tem/main.cpp.o.d -o CMakeFiles/xy_form_tem.dir/xy-form-tem/main.cpp.o -c /mnt/nvme0n1p1/honours-project/c-attempt/xy-form-tem/main.cpp

CMakeFiles/xy_form_tem.dir/xy-form-tem/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/xy_form_tem.dir/xy-form-tem/main.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/nvme0n1p1/honours-project/c-attempt/xy-form-tem/main.cpp > CMakeFiles/xy_form_tem.dir/xy-form-tem/main.cpp.i

CMakeFiles/xy_form_tem.dir/xy-form-tem/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/xy_form_tem.dir/xy-form-tem/main.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/nvme0n1p1/honours-project/c-attempt/xy-form-tem/main.cpp -o CMakeFiles/xy_form_tem.dir/xy-form-tem/main.cpp.s

# Object files for target xy_form_tem
xy_form_tem_OBJECTS = \
"CMakeFiles/xy_form_tem.dir/xy-form-tem/main.cpp.o"

# External object files for target xy_form_tem
xy_form_tem_EXTERNAL_OBJECTS =

/mnt/nvme0n1p1/honours-project/c-attempt/xy_form_tem: CMakeFiles/xy_form_tem.dir/xy-form-tem/main.cpp.o
/mnt/nvme0n1p1/honours-project/c-attempt/xy_form_tem: CMakeFiles/xy_form_tem.dir/build.make
/mnt/nvme0n1p1/honours-project/c-attempt/xy_form_tem: /usr/lib64/libomp.so
/mnt/nvme0n1p1/honours-project/c-attempt/xy_form_tem: /usr/lib64/libpthread.so
/mnt/nvme0n1p1/honours-project/c-attempt/xy_form_tem: CMakeFiles/xy_form_tem.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/nvme0n1p1/honours-project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable /mnt/nvme0n1p1/honours-project/c-attempt/xy_form_tem"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/xy_form_tem.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/xy_form_tem.dir/build: /mnt/nvme0n1p1/honours-project/c-attempt/xy_form_tem
.PHONY : CMakeFiles/xy_form_tem.dir/build

CMakeFiles/xy_form_tem.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/xy_form_tem.dir/cmake_clean.cmake
.PHONY : CMakeFiles/xy_form_tem.dir/clean

CMakeFiles/xy_form_tem.dir/depend:
	cd /mnt/nvme0n1p1/honours-project/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/nvme0n1p1/honours-project/c-attempt /mnt/nvme0n1p1/honours-project/c-attempt /mnt/nvme0n1p1/honours-project/build /mnt/nvme0n1p1/honours-project/build /mnt/nvme0n1p1/honours-project/build/CMakeFiles/xy_form_tem.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/xy_form_tem.dir/depend
