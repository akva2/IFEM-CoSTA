project(CoSTA)

cmake_minimum_required(VERSION 2.6)

set(IFEM_INSTALL_DOXY 0)

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

set(COSTA_SRCS )

set(ELASTICITY_DIR ${PROJECT_SOURCE_DIR}/../IFEM-Elasticity)
if(NOT EXISTS ${ELASTICITY_DIR})
  set(ELASTICITY_DIR ${PROJECT_SOURCE_DIR}/../Elasticity)
endif()

set(THERMO_DIR ${PROJECT_SOURCE_DIR}/../IFEM-ThermoElasticity)
if(NOT EXISTS ${THERMO_DIR})
  set(THERMO_DIR ${PROJECT_SOURCE_DIR}/../ThermoElasticity)
endif()

set(AD_DIR ${PROJECT_SOURCE_DIR}/../IFEM-AdvectionDiffusion)
if(NOT EXISTS ${AD_DIR})
  set(AD_DIR ${PROJECT_SOURCE_DIR}/../AdvectionDiffusion)
endif()

set(DARCY_DIR ${PROJECT_SOURCE_DIR}/../IFEM-Darcy)
if(NOT EXISTS ${DARCY_DIR})
  set(DARCY_DIR ${PROJECT_SOURCE_DIR}/../Darcy)
endif()

include_directories(${IFEM_INCLUDES} ${PROJECT_SOURCE_DIR} ../Common
                    ${THERMO_DIR}/ThermoElastic ${ELASTICITY_DIR}
                    ${AD_DIR}/Common/AdvectionDiffusion
                    ${DARCY_DIR}/Common/Darcy)

add_subdirectory(${AD_DIR}/Common/AdvectionDiffusion AdvectionDiffusion)
add_subdirectory(${DARCY_DIR}/Common/Darcy Darcy)
add_subdirectory(${ELASTICITY_DIR} Elastic)
add_subdirectory(${THERMO_DIR}/ThermoElastic ThermoElastic)

set(LIBRARY_OUTPUT_PATH IFEM_CoSTA)

pybind11_add_module(IFEM_CoSTA IFEM_CoSTA.C)
target_sources(IFEM_CoSTA PRIVATE CoSTA_AdvectionDiffusion.C
                                  CoSTA_Darcy.C
                                  CoSTA_HeatEquation.C
                                  CoSTAModule.h)

target_link_libraries(IFEM_CoSTA PRIVATE -Wl,--as-needed CommonDarcy CommonAD Elasticity ThermoElastic IFEMAppCommon IFEM -Wl,--no-as-needed pybind11::pybind11 Python3::Python)

# For generating the doxy
set(EXTRA_DOXY_PATHS "${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR}")
add_doc_target(CoSTA CoSTA)

file(WRITE ${PROJECT_BINARY_DIR}/IFEM_CoSTA/__init__.py "from IFEM_CoSTA.IFEM_CoSTA import *\n")
file(GENERATE OUTPUT ${PROJECT_BINARY_DIR}/setup.py
     INPUT ${PROJECT_SOURCE_DIR}/setup.py.in)

install(CODE "message(\"Please use pip to install the package from build directory\")")
