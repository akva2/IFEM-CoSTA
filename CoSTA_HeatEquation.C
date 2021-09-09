// $Id$
//==============================================================================
//!
//! \file CoSTA_HeatEquation.C
//!
//! \date Sep 9 2021
//!
//! \author Arne Morten Kvarving / SINTEF
//!
//! \brief Exports the HeatEquation solver to the IFEM_CoSTA module.
//!
//==============================================================================

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "CoSTAModule.h"
#include "HeatEquation.h"
#include "SIMHeatEquation.h"

#include "SIMconfigure.h"


template<class Dim> using SIMHeatEq = SIMHeatEquation<Dim,HeatEquation>; //!< One-parameter type alias for SIMHeatEquation


//! \brief Specialization for SIMHeatEq.
template<>
struct CoSTASIMAllocator<SIMHeatEq> {
  //! \brief Method to allocate a given dimensionality of a SIMHeatEq.
  //! \param newModel Simulator to allocate
  //! \param model Pointer to SIMbase interface for simulator
  //! \param solModel Pointer to SIMsolution interface for simulator
  //! \param infile Input file to parse.
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
