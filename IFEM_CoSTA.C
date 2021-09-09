#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "CoSTAModule.h"
#include "HeatEquation.h"
#include "SIMHeatEquation.h"


Profiler prof("CoSTA-Module");

void export_AdvectionDiffusion(pybind11::module& m);
void export_HeatEquation(pybind11::module& m);


PYBIND11_MODULE(IFEM_CoSTA, m)
{
  export_AdvectionDiffusion(m);
  export_HeatEquation(m);
}
