project(CoSTA)

cmake_minimum_required(VERSION 2.6)

# Add local modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      ${PROJECT_SOURCE_DIR}/../../cmake/Modules)

if(NOT IFEM_CONFIGURED)
  find_package(IFEM REQUIRED)
  include_directories(${IFEM_INCLUDE_DIRS})
  add_definitions(${IFEM_DEFINITIONS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${IFEM_CXX_FLAGS}")
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(Python3 COMPONENTS Development)
find_package(pybind11 REQUIRED)

set(AD_DIR ${PROJECT_SOURCE_DIR}/../IFEM-AdvectionDiffusion)
if(NOT EXISTS ${AD_DIR})
  set(AD_DIR ${PROJECT_SOURCE_DIR}/../AdvectionDiffusion)
endif()

set(DARCY_DIR ${PROJECT_SOURCE_DIR}/../IFEM-Darcy)
if(NOT EXISTS ${DARCY_DIR})
  set(DARCY_DIR ${PROJECT_SOURCE_DIR}/../Darcy)
endif()

include(IFEMTesting)

include_directories(${IFEM_INCLUDES} ${PROJECT_SOURCE_DIR} ../Common
                    ${AD_DIR}/Common/AdvectionDiffusion
                    ${DARCY_DIR}/Common/Darcy)

add_subdirectory(${AD_DIR}/Common/AdvectionDiffusion AdvectionDiffusion)
add_subdirectory(${DARCY_DIR}/Common/Darcy Darcy)

set(LIBRARY_OUTPUT_PATH IFEM_CoSTA)

pybind11_add_module(IFEM_CoSTA IFEM_CoSTA.C)
target_sources(IFEM_CoSTA PRIVATE CoSTA_AdvectionDiffusion.C
                                  CoSTA_Darcy.C
                                  CoSTA_HeatEquation.C
                                  CoSTAModule.h)

target_link_libraries(IFEM_CoSTA PRIVATE -Wl,--as-needed CommonDarcy CommonAD IFEMAppCommon ${IFEM_LIBRARIES} -Wl,--no-as-needed pybind11::pybind11 Python3::Python)

# For generating the doxy
set(EXTRA_DOXY_PATHS "${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR}")
add_doc_target(CoSTA CoSTA)

# Tests
list(APPEND TEST_APPS IFEM_CoSTA)
include(CTest)

find_program(PYTEST_COMMAND NAMES pytest-3 pytest)

add_test(NAME python_tests COMMAND ${PYTEST_COMMAND} --junitxml=${PROJECT_BINARY_DIR}/pytestoutput.xml WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/Test)
set_tests_properties(python_tests PROPERTIES ENVIRONMENT PYTHONPATH=${PROJECT_BINARY_DIR})

get_target_property(COSTA_SOURCES IFEM_CoSTA SOURCES)
foreach(SRC ${COSTA_SOURCES})
  list(APPEND CHECK_SOURCES ${PROJECT_SOURCE_DIR}/${SRC})
endforeach()

# Installation etc
file(WRITE ${PROJECT_BINARY_DIR}/IFEM_CoSTA/__init__.py "from IFEM_CoSTA.IFEM_CoSTA import *\n")
file(GENERATE OUTPUT ${PROJECT_BINARY_DIR}/setup.py
     INPUT ${PROJECT_SOURCE_DIR}/setup.py.in)

install(CODE "message(\"Please use pip to install the package from build directory\")")

if(IFEM_COMMON_APP_BUILD)
  set(TEST_APPS ${TEST_APPS} PARENT_SCOPE)
  set(UNIT_TEST_NUMBER ${UNIT_TEST_NUMBER} PARENT_SCOPE)
else()
  add_check_target()
endif()
