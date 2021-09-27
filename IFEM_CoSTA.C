// $Id$
//==============================================================================
//!
//! \file IFEM_CoSTA.C
//!
//! \date Sep 9 2021
//!
//! \author Arne Morten Kvarving / SINTEF
//!
//! \brief Exports the IFEM_CoSTA python module.
//!
//==============================================================================

#include "Profiler.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>


//! \brief Exports the AdvectionDiffusion CoSTA module.
void export_AdvectionDiffusion(pybind11::module& m);

//! \brief Exports the Darcy CoSTA module.
void export_Darcy(pybind11::module& m);

//! \brief Exports the AdvectionDiffusion CoSTA module.
void export_HeatEquation(pybind11::module& m);


Profiler prof("CoSTA-Module", false); //!< Global instance of profiler


//! \brief Exports the IFEM_CoSTA python module.
PYBIND11_MODULE(IFEM_CoSTA, m)
{
  export_AdvectionDiffusion(m);
  export_Darcy(m);
  export_HeatEquation(m);
}
