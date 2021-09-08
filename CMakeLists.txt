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

include_directories(${IFEM_INCLUDES} ${PROJECT_SOURCE_DIR} ../Common
                    ${THERMO_DIR}/ThermoElastic ${ELASTICITY_DIR})

add_subdirectory(${ELASTICITY_DIR} Elastic)
add_subdirectory(${THERMO_DIR}/ThermoElastic ThermoElastic)

set(LIBRARY_OUTPUT_PATH CoSTA_HeatEquation)

pybind11_add_module(CoSTA_HeatEquation CoSTA-HeatEquation.C)
target_sources(CoSTA_HeatEquation PRIVATE CoSTAModule.h)

target_link_libraries(CoSTA_HeatEquation PRIVATE -Wl,--as-needed Elasticity ThermoElastic IFEMAppCommon IFEM -Wl,--no-as-needed pybind11::pybind11 Python3::Python)
