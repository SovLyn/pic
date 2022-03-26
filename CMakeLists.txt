cmake_minimum_required(VERSION 3.5)

# Set a default build type if none was specified
# Try cmake {{path/to/project_directory}} -D {{CMAKE_BUILD_TYPE=Release}}
#  * Release - Adds the `-O3 -DNDEBUG` flags to the compiler
#  * Debug - Adds the `-g` flag
#  * MinSizeRel - Adds `-Os -DNDEBUG`
#  * RelWithDebInfo - Adds `-O2 -g -DNDEBUG` flags

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message("Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

project(cufluid LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS )

if(CUDA_FOUND)
	message ("CUDA found")
else()
	message (FATAL_ERROR, "Cannot find CUDA")
endif()

find_package(Boost REQUIRED COMPONENTS filesystem system)

# check if boost was found
if(Boost_FOUND)
    message ("boost found")
else()
    message (FATAL_ERROR "Cannot find Boost")
endif()

find_package(OpenGL REQUIRED)

if(OpenGL_FOUND)
	message ("OpenGL found")
else()
	message (FATAL_ERROR "Cannot find OpenGL")
endif()

find_package(GLUT REQUIRED)

if(GLUT_FOUND)
	message ("GLUT found")
else()
	message (FATAL_ERROR "Cannot find GLUT")
endif()

find_package(glfw3 REQUIRED)

if(glfw3_FOUND)
	message ("GLFW3 found")
else()
	message (FATAL_ERROR "Cannot find GLFW3")
endif()

include_directories(${OPENGL_INCLUDE_DIRS}
	${GLUT_INCLUDE_DIRS}
	${PROJECT_SOURCE_DIR}/include)

add_library(glprocess
	SHARED
	src/glprocess.cu)

target_include_directories(glprocess
	PUBLIC
	# All the directories following PUBLIC 
	# will be used for the current target and 
	# the other targets that have dependencies 
	# on the current target.
	${PROJECT_SOURCE_DIR}/include/)

add_library(MAC
	SHARED
	src/MAC.cu)

target_include_directories(MAC
	PUBLIC
	${PROJECT_SOURCE_DIR}/include/)

add_executable(${PROJECT_NAME}
	src/main.cu)

target_link_libraries(${PROJECT_NAME}
	PRIVATE
	glprocess)

target_link_libraries(${PROJECT_NAME}
	PRIVATE
	MAC)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
	Boost::boost OpenGL glut glfw)
