#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "CoSTAModule.h"
#include "HeatEquation.h"
#include "SIMHeatEquation.h"

#include "SIMconfigure.h"


template<class Dim> using SIMHeatEq = SIMHeatEquation<Dim,HeatEquation>;


template<>
struct CoSTASIMAllocator<SIMHeatEq> {
   template<class Dim>
   void allocate(std::unique_ptr<SIMHeatEq<Dim>>& newModel, SIMbase*& model,
                 SIMsolution*& solModel, const std::string& infile)
   {
      newModel = std::make_unique<SIMHeatEq<Dim>>(1);
      model = newModel.get();
      solModel = newModel.get();
      if (ConfigureSIM(*newModel, const_cast<char*>(infile.c_str())))
        throw std::runtime_error("Error reading input file");
   }
};


void export_HeatEquation(pybind11::module& m)
{
  pybind11::class_<CoSTAModule<SIMHeatEq>>(m, "HeatEquation")
  .def(pybind11::init<const std::string&>())
  .def("correct", &CoSTAModule<SIMHeatEq>::correct)
  .def("predict", &CoSTAModule<SIMHeatEq>::predict)
  .def("residual", &CoSTAModule<SIMHeatEq>::residual)
  .def_readonly("ndof", &CoSTAModule<SIMHeatEq>::ndof);
}
