cmake_minimum_required(VERSION 3.18)
project(multiverse_native LANGUAGES CXX)

option(MULTIVERSE_BUILD_NATIVE "Build the optional Multiverse C++ simulation extension" ON)
if(NOT MULTIVERSE_BUILD_NATIVE)
  return()
endif()

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import pathlib, mujoco; p=pathlib.Path(mujoco.__file__).parent/'include'; candidates=[p,p/'mujoco']; print(next(str(x) for x in candidates if (x/'mujoco'/'mujoco.h').is_file()))"
  RESULT_VARIABLE MUJOCO_PYTHON_RESULT
  OUTPUT_VARIABLE MUJOCO_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(NOT MUJOCO_PYTHON_RESULT EQUAL 0 OR NOT EXISTS "${MUJOCO_INCLUDE_DIR}/mujoco/mujoco.h")
  message(FATAL_ERROR "Install the optional MuJoCo Python wheel before configuring the native runtime")
endif()

pybind11_add_module(_multiverse_native MODULE src/mujoco_batch.cpp)
target_compile_features(_multiverse_native PRIVATE cxx_std_17)
target_include_directories(_multiverse_native PRIVATE "${MUJOCO_INCLUDE_DIR}")
set_target_properties(_multiverse_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_target_properties(_multiverse_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}")
endforeach()
if(MSVC)
  target_compile_options(_multiverse_native PRIVATE /W4 /permissive-)
else()
  target_compile_options(_multiverse_native PRIVATE -Wall -Wextra -Wpedantic)
  target_link_libraries(_multiverse_native PRIVATE ${CMAKE_DL_LIBS})
endif()

# The MuJoCo Python wheel provides the runtime library and C headers. We load
# the shared library at runtime, so this extension does not depend on a
# platform-specific import library being present in the wheel.
install(TARGETS _multiverse_native LIBRARY DESTINATION . RUNTIME DESTINATION .)
