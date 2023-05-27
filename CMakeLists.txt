# CMakeList.txt: cuda_practice 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

project (cuda_practice)
enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# 将源代码添加到此项目的可执行文件。
add_executable (${PROJECT_NAME} main.cu)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET cuda_practice PROPERTY CXX_STANDARD 20)
endif()

# TODO: 如有需要，请添加测试并安装目标。
add_library(Toolbox STATIC 
  vec3double.cu
  common.cu
  ray.cu
  camera.cu
  sphere.cu
)


target_link_libraries(${PROJECT_NAME} Toolbox)
target_link_libraries(${PROJECT_NAME} Threads::Threads)
set_target_properties(Toolbox PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
