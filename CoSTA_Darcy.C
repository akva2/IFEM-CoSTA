// $Id$
//==============================================================================
//!
//! \file CoSTA_Darcy.C
//!
//! \date Sep 9 2021
//!
//! \author Arne Morten Kvarving / SINTEF
//!
//! \brief Exports the Darcy solver to the IFEM_CoSTA module.
//!
//==============================================================================

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "CoSTAModule.h"
#include "SIMDarcy.h"

#include "AlgEqSystem.h"
#include "ElmMats.h"
#include "SIMconfigure.h"
#include "SystemMatrix.h"


/*!
 \brief Integrand for Darcy with CoSTA additions.
*/

class DarcyCoSTA : public Darcy
{
public:
  //! \brief The default constructor initializes all pointers to zero.
  //! \param[in] n Number of spatial dimensions
  DarcyCoSTA(unsigned short int n) :
    Darcy(n, 1)
  {
  }

  using Darcy::finalizeElement;
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
 \brief CoSTA simulator for Darcy.
*/

template<class Dim>
class SIMDarcyCoSTA : public SIMDarcy<Dim>,
                      public CoSTASIMHelper
{
public:
  //! \brief Constructor.
  //! \param integrand Reference to integrand to use
  SIMDarcyCoSTA(DarcyCoSTA& integrand) :
    SIMDarcy<Dim>(integrand)
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


//! \brief Specialization for SIMDarcy.
template<>
struct CoSTASIMAllocator<SIMDarcyCoSTA> {
  //! \brief Method to allocate a given dimensionality of a SIMHeatEq.
  //! \param newModel Simulator to allocate
  //! \param model Pointer to SIMbase interface for simulator
  //! \param solModel Pointer to SIMsolution interface for simulator
  //! \param infile Input file to parse.
  template<class Dim>
  void allocate(std::unique_ptr<SIMDarcyCoSTA<Dim>>& newModel, SIMbase*& model,
                SIMsolution*& solModel, const std::string& infile)
  {
    integrand = std::make_unique<DarcyCoSTA>(Dim::dimension);
    newModel = std::make_unique<SIMDarcyCoSTA<Dim>>(*integrand);
    model = newModel.get();
    solModel = newModel.get();
    if (ConfigureSIM(static_cast<SIMDarcy<Dim>&>(*newModel),
                     const_cast<char*>(infile.c_str())))
      throw std::runtime_error("Error reading input file");
  }

  std::unique_ptr<DarcyCoSTA> integrand; //!< Pointer to integrand instance
};


void export_Darcy(pybind11::module& m)
{
  CoSTAModule<SIMDarcyCoSTA>::pyExport(m, "Darcy");
}
