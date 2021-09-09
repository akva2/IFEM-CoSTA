#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "CoSTAModule.h"
#include "HeatEquation.h"
#include "SIMHeatEquation.h"


void export_HeatEquation(pybind11::module& m);


PYBIND11_MODULE(IFEM_CoSTA, m)
{
  export_HeatEquation(m);
}
