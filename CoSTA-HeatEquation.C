#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "CoSTAModule.h"
#include "HeatEquation.h"
#include "SIMHeatEquation.h"

template<class Dim> using SIMHeatEq = SIMHeatEquation<Dim,HeatEquation>;


PYBIND11_MODULE(CoSTA_HeatEquation, m) {
  pybind11::class_<CoSTAModule<SIMHeatEq>>(m, "CoSTA_HeatEquation")
  .def(pybind11::init<const std::string&>())
  .def("correct", &CoSTAModule<SIMHeatEq>::correct)
  .def("predict", &CoSTAModule<SIMHeatEq>::predict)
  .def("residual", &CoSTAModule<SIMHeatEq>::residual)
  .def_readonly("ndof", &CoSTAModule<SIMHeatEq>::ndof);
}
