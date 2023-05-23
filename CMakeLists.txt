# CMakeList.txt: cuda_practice 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

project (cuda_practice)
enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

# 将源代码添加到此项目的可执行文件。
add_executable (${PROJECT_NAME} cuda_practice.cu)
add_executable (test test.cu)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET cuda_practice PROPERTY CXX_STANDARD 20)
endif()

# TODO: 如有需要，请添加测试并安装目标。
add_library(Toolbox STATIC 
  common.cu 
  vec3double.cu 
  camera.cu 
  shape.cu
  )

set_target_properties(test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME} Toolbox)
set_target_properties(Toolbox PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
