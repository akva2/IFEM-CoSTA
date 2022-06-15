// $Id$
//==============================================================================
//!
//! \file CoSTA_AdvectionDiffusion.C
//!
//! \date Sep 9 2021
//!
//! \author Arne Morten Kvarving / SINTEF
//!
//! \brief Exports the AdvectionDiffusion solver to the IFEM_CoSTA module.
//!
//==============================================================================

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "CoSTAModule.h"

#include "AlgEqSystem.h"
#include "ExprFunctions.h"
#include "SystemMatrix.h"

#include "AdvectionDiffusionBDF.h"
#include "SIMAD.h"


/*!
 \brief Integrand for AdvectionDiffusion with CoSTA additions.
*/

class AdvectionDiffusionCoSTA : public AdvectionDiffusionBDF
{
public:
  //! \brief The default constructor initializes all pointers to zero.
  //! \param[in] n Number of spatial dimensions
  explicit AdvectionDiffusionCoSTA(unsigned short int n) :
    AdvectionDiffusionBDF(n, TimeIntegration::BE)
  {
  }

  using AdvectionDiffusionBDF::finalizeElement;
  //! \brief Finalizes the element quantities after the numerical integration.
  //! \details This method is invoked once for each element, after the numerical
  //! integration loop over interior points is finished and before the resulting
  //! element quantities are assembled into their system level equivalents.
  //! It is used here to calculate the linear residual if requested.
  bool finalizeElement(LocalIntegral& elmInt) override
  {
    if (m_mode == SIM::RHS_ONLY) {
      ElmMats& A = static_cast<ElmMats&>(elmInt);
      A.A[0].multiply(A.vec[0], A.b[0], 1.0, -1.0);
    }

    return true;
  }
};


/*!
 \brief CoSTA simulator for AdvectionDiffusion.
*/

template<class Dim>
class SIMADCoSTA : public SIMAD<Dim,AdvectionDiffusionBDF>,
                   public CoSTASIMHelper
{
public:
  //! \brief Constructor
  //! \param ad The integrand to use
  explicit SIMADCoSTA(AdvectionDiffusionCoSTA& ad) :
    SIMAD<Dim,AdvectionDiffusionBDF>(ad,true)
  {
  }

  //! \brief Currently unused.
  void setParam(const std::string&, double) {}

  //! \brief Returns analytical solutions projected on primary basis.
  //! \param t Time to evaluate at
  std::map<std::string, std::vector<double>> getAnaSols(double t)
  {
    return this->CoSTASIMHelper::getAsolScalar(t, this->mySol, this);
  }

  //! \brief Returns a quantity of interest.
  RealArray getQI(const RealArray&,
                  const TimeDomain&,
                  const std::string&)
  {
    return RealArray();
  }

protected:
  //! \brief Assembles problem-dependent discrete terms, if any.
  bool assembleDiscreteTerms(const IntegrandBase*,
                             const TimeDomain&) override
  {
    return this->assembleDiscreteLoad(this->getNoDOFs(),
                                      Dim::mySam,
                                      Dim::myEqSys->getVector(0));
  }
};


//! \brief Specialization for SIMADCoSTA.
template<>
struct CoSTASIMAllocator<SIMADCoSTA> {
  //! \brief Method to allocate a given dimensionality of a SIMAdv.
  //! \param newModel Simulator to allocate
  //! \param model Pointer to SIMbase interface for simulator
  //! \param solModel Pointer to SIMsolution interface for simulatorA
  //! \param infile Input file to parse.
  template<class Dim>
  void allocate(std::unique_ptr<SIMADCoSTA<Dim>>& newModel, SIMbase*& model,
                SIMsolution*& solModel, const std::string& infile)
  {
    integrand = std::make_unique<AdvectionDiffusionCoSTA>(Dim::dimension);
    newModel = std::make_unique<SIMADCoSTA<Dim>>(*integrand);
    model = newModel.get();
    solModel = newModel.get();
    if (ConfigureSIM(static_cast<SIMAD<Dim,AdvectionDiffusionBDF>&>(*newModel),
                     const_cast<char*>(infile.c_str())))
      throw std::runtime_error("Error reading input file");
  }

   std::unique_ptr<AdvectionDiffusionCoSTA> integrand; //!< Pointer to integrand instance
};


void export_AdvectionDiffusion(pybind11::module& m)
{
  CoSTAModule<SIMADCoSTA>::pyExport(m, "AdvectionDiffusion");
}
