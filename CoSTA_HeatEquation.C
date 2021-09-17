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

#include "AlgEqSystem.h"
#include "ElmMats.h"
#include "ExprFunctions.h"
#include "SIMconfigure.h"
#include "SystemMatrix.h"


/*!
 \brief Integrand for HeatEquation with CoSTA additions.
*/

class HeatEquationCoSTA : public HeatEquation
{
public:
  //! \brief Constructor.
  //! \param n Number of spatial dimensions
  //! \param torder Time integration order
  explicit HeatEquationCoSTA(unsigned short int n, int torder) :
    HeatEquation(n,torder)
  {
  }

  using HeatEquation::finalizeElement;
  //! \brief Finalizes the element quantities after the numerical integration.
  //! \details This method is invoked once for each element, after the numerical
  //! integration loop over interior points is finished and before the resulting
  //! element quantities are assembled into their system level equivalents.
  //! It is used here to calculate the linear residual if requested.
  bool finalizeElement (LocalIntegral& elmInt) override
  {
    if (m_mode == SIM::RHS_ONLY) {
      ElmMats& A = static_cast<ElmMats&>(elmInt);
      A.A[0].multiply(A.vec[0], A.b[0], -1.0, 1.0);
    }

    return true;
  }

  //! \brief Set a parameter in the source and flux functions.
  //! \param name Name of parameter
  //! \param value Value of parameter
  void setParam(const std::string& name, double value)
  {
    EvalFunction* f = dynamic_cast<EvalFunction*>(sourceTerm);
    if (f)
      f->setParam(name, value);
    f = dynamic_cast<EvalFunction*>(flux);
    if (f)
      f->setParam(name, value);
  }
};


/*!
 \brief CoSTA simulator for HeatEquation.
*/

template<class Dim>
class SIMHeatCoSTA : public SIMHeatEquation<Dim,HeatEquationCoSTA>,
                     public CoSTASIMHelper
{
public:
  //! \brief Constructor
  //! \param torder Time integration order
  SIMHeatCoSTA(int torder) : SIMHeatEquation<Dim,HeatEquationCoSTA>(torder)
  {
  }

  //! \brief Set a parameter in relevant functions.
  //! \param name Name of parameter
  //! \param value Value of parameter
  void setParam(const std::string& name, double value)
  {
    this->heq.setParam(name, value);
    if (this->mySol) {
      EvalFunction* f = dynamic_cast<EvalFunction*>(this->mySol->getScalarSol());
      if (f)
        f->setParam(name, value);
      VecFuncExpr* v = dynamic_cast<VecFuncExpr*>(this->mySol->getScalarSecSol());
      if (v)
        v->setParam(name, value);
    }
  }

protected:
  //! \brief Assembles problem-dependent discrete terms, if any.
  bool assembleDiscreteTerms(const IntegrandBase*,
                             const TimeDomain&) override
  {
    return this->assembleDiscreteLoad(this->getNoDOFs(),
                                      this->mySam,
                                      this->myEqSys->getVector(0));
  }
};


//! \brief Specialization for SIMHeatCoSTA.
template<>
struct CoSTASIMAllocator<SIMHeatCoSTA> {
  //! \brief Method to allocate a given dimensionality of a SIMHeatEq.
  //! \param newModel Simulator to allocate
  //! \param model Pointer to SIMbase interface for simulator
  //! \param solModel Pointer to SIMsolution interface for simulator
  //! \param infile Input file to parse.
  template<class Dim>
  void allocate(std::unique_ptr<SIMHeatCoSTA<Dim>>& newModel, SIMbase*& model,
                SIMsolution*& solModel, const std::string& infile)
  {
    newModel = std::make_unique<SIMHeatCoSTA<Dim>>(1);
    model = newModel.get();
    solModel = newModel.get();
    if (ConfigureSIM(static_cast<SIMHeatEquation<Dim,HeatEquationCoSTA>&>(*newModel),
                     const_cast<char*>(infile.c_str())))
      throw std::runtime_error("Error reading input file");
  }
};


void export_HeatEquation(pybind11::module& m)
{
  CoSTAModule<SIMHeatCoSTA>::pyExport(m, "HeatEquation");
}
