#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "CoSTAModule.h"
#include "HeatEquation.h"
#include "SIMHeatEquation.h"


template<class Dim> using SIMHeatEq = SIMHeatEquation<Dim,HeatEquation>;


void export_HeatEquation(pybind11::module& m)
{
  pybind11::class_<CoSTAModule<SIMHeatEq>>(m, "HeatEquation")
  .def(pybind11::init<const std::string&>())
  .def("correct", &CoSTAModule<SIMHeatEq>::correct)
  .def("predict", &CoSTAModule<SIMHeatEq>::predict)
  .def("residual", &CoSTAModule<SIMHeatEq>::residual)
  .def_readonly("ndof", &CoSTAModule<SIMHeatEq>::ndof);
}
