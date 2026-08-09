cmake_minimum_required(VERSION 3.22)

project(CoSTA LANGUAGES C CXX)

cmake_policy(SET CMP0057 NEW)

# SKBUILD holds scikit-build-core's protocol version (e.g. "2"), not a boolean,
# so it can't be passed as option()'s default value directly.
if(DEFINED SKBUILD)
  set(COSTA_BUILD_PYTHON_PACKAGE_DEFAULT ON)
else()
  set(COSTA_BUILD_PYTHON_PACKAGE_DEFAULT OFF)
endif()
option(COSTA_BUILD_PYTHON_PACKAGE
       "Install the Python IFEM_CoSTA package layout (for building wheels via pip)"
       ${COSTA_BUILD_PYTHON_PACKAGE_DEFAULT})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT TARGET IFEM)
  if(IFEM_AS_SUBMODULE)
    add_subdirectory(../.. IFEM)
    set(IFEM_PATH ${PROJECT_SOURCE_DIR}/../..)
  else()
    get_filename_component(BUILD_DIR ${PROJECT_BINARY_DIR} NAME)
    set(IFEM_DIR ${PROJECT_SOURCE_DIR}/../../${BUILD_DIR})
    find_package(IFEM REQUIRED)
  endif()
endif()

find_package(Python3 COMPONENTS Development Interpreter)
find_package(pybind11 REQUIRED)

ifem_add_dependency_dir(
  APP
    AdvectionDiffusion
  DIR
    AdvectionDiffusion
  TARGET
    IFEM::AdvectionDiffusion
)
ifem_add_dependency_dir(
  APP
    Darcy
  DIR
    Darcy
  TARGET
    IFEM::Darcy
)

set(LIBRARY_OUTPUT_PATH IFEM_CoSTA)

file(COPY ${PROJECT_SOURCE_DIR}/src/IFEM_CoSTA/__init__.py
     DESTINATION ${PROJECT_BINARY_DIR}/IFEM_CoSTA)

pybind11_add_module(IFEM_CoSTA IFEM_CoSTA.C)
target_sources(IFEM_CoSTA PRIVATE
  CoSTA_AdvectionDiffusion.C
  CoSTA_Darcy.C
  CoSTA_HeatEquation.C
)
if(CMAKE_VERSION GREATER_EQUAL 3.23)
  target_sources(IFEM_CoSTA PUBLIC
    FILE_SET HEADERS
    FILES
      CoSTAModule.h
  )
endif()

target_link_libraries(IFEM_CoSTA
  PRIVATE
    IFEM::AdvectionDiffusion
    IFEM::Darcy
    pybind11::pybind11
    Python3::Python
)

# Tests
enable_testing()

list(APPEND TEST_APPS IFEM_CoSTA)

find_program(PYTEST_COMMAND NAMES pytest-3 pytest)

add_test(
    NAME
      python_tests
    COMMAND
      ${PYTEST_COMMAND} --junitxml=${PROJECT_BINARY_DIR}/pytestoutput.xml
    WORKING_DIRECTORY
      ${PROJECT_SOURCE_DIR}/Test
)
set_tests_properties(
  python_tests
  PROPERTIES
    ENVIRONMENT PYTHONPATH=${PROJECT_BINARY_DIR}
)

if(IFEM_COMMON_APP_BUILD)
  set(TEST_APPS ${TEST_APPS} PARENT_SCOPE)
else()
  ifem_add_check_target()
endif()

# For generating the doxy
ifem_add_doc_target(
  TARGET
    CoSTA
  DEPENDENCIES
    AdvectionDiffusion
    Darcy
)

# Installation etc
if(COSTA_BUILD_PYTHON_PACKAGE)
  install(TARGETS IFEM_CoSTA DESTINATION IFEM_CoSTA)
  install(FILES ${PROJECT_SOURCE_DIR}/src/IFEM_CoSTA/__init__.py DESTINATION IFEM_CoSTA)
else()
  install(CODE "message(\"Configure with -DCOSTA_BUILD_PYTHON_PACKAGE=ON, or use pip (see pyproject.toml), to install the Python package.\")")
endif()
