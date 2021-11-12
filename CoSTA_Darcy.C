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
#include "Darcy.h"
#include "MixedDarcy.h"
#include "SIMDarcy.h"

#include "AlgEqSystem.h"
#include "ASMmxBase.h"
#include "ElmMats.h"
#include "ExprFunctions.h"
#include "SIMconfigure.h"
#include "SystemMatrix.h"
#include "TimeIntUtils.h"
#include "Utilities.h"

#include <tinyxml.h>


/*!
 * \brief Preparsing to check the Darcy formulation to use.
*/

class DarcyPreParse : public XMLInputBase
{
public:
  bool twofield = false; //!< True to use a two-field formulation
  int torder = 0; //!< Time integration order

  //! \brief Constructor.
  DarcyPreParse(const std::string& file)
  {
    this->readXML(file.c_str(), false);
  }

protected:
  //! \brief Parse an XML element.
  bool parse(const TiXmlElement* elem)
  {
    if (!strcasecmp(elem->Value(),"timestepping")) {
      std::string type;
      if (utl::getAttribute(elem,"type",type))
        torder = TimeIntegration::Order(TimeIntegration::get(type));
    } else if (!strcasecmp(elem->Value(),"darcy")) {
      utl::getAttribute(elem,"twofield",twofield);
      ASMmxBase::Type = ASMmxBase::NONE;
      const char* formulation = elem->Attribute("formulation");
      if (formulation) {
        if (strcasecmp(formulation, "th" ) == 0 ||
            strcasecmp(formulation, "mixed") == 0) {
          ASMmxBase::Type = ASMmxBase::REDUCED_CONT_RAISE_BASIS2;
          twofield = true;
        } else if (strcasecmp(formulation,"frth") == 0 ||
                   strcasecmp(formulation,"mixed_full") == 0) {
          ASMmxBase::Type = ASMmxBase::FULL_CONT_RAISE_BASIS2;
          twofield = true;
        }
    }
  }

  return true;
}

};


/*!
 \brief Integrand for Darcy with CoSTA additions.
*/

class DarcyCoSTA : public Darcy
{
public:
  //! \brief The default constructor initializes all pointers to zero.
  //! \param[in] n Number of spatial dimensions
  explicit DarcyCoSTA(unsigned short int n, int torder) :
    Darcy(n, torder)
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

  //! \brief Set a parameter in the source and flux functions.
  //! \param name Name of parameter
  //! \param value Value of parameter
  void setParam(const std::string& name, double value) override
  {
    VecFuncExpr* f = dynamic_cast<VecFuncExpr*>(permvalues);
    if (f)
      f->setParam(name, value);

    EvalFunction* fs = dynamic_cast<EvalFunction*>(source);
    if (fs)
      fs->setParam(name,value);
    fs = dynamic_cast<EvalFunction*>(porosity);
    if (fs)
      fs->setParam(name,value);
    fs = dynamic_cast<EvalFunction*>(dispersivity);
    if (fs)
      fs->setParam(name,value);
    fs = dynamic_cast<EvalFunction*>(source);
    if (fs)
      fs->setParam(name,value);
  }
};


/*!
 \brief Integrand for two-field Darcy with CoSTA additions.
*/

class MixedDarcyCoSTA : public MixedDarcy
{
public:
  //! \brief The default constructor initializes all pointers to zero.
  //! \param[in] n Number of spatial dimensions
  explicit MixedDarcyCoSTA(unsigned short int n, int torder) :
    MixedDarcy(n, torder)
  {
  }

  //! \brief Set a parameter in the source and flux functions.
  //! \param name Name of parameter
  //! \param value Value of parameter
  void setParam(const std::string& name, double value) override
  {
    VecFuncExpr* f = dynamic_cast<VecFuncExpr*>(permvalues);
    if (f)
      f->setParam(name, value);

    EvalFunction* fs = dynamic_cast<EvalFunction*>(source);
    if (fs)
      fs->setParam(name,value);
    fs = dynamic_cast<EvalFunction*>(porosity);
    if (fs)
      fs->setParam(name,value);
    fs = dynamic_cast<EvalFunction*>(dispersivity);
    if (fs)
      fs->setParam(name,value);
    fs = dynamic_cast<EvalFunction*>(source);
    if (fs)
      fs->setParam(name,value);
    fs = dynamic_cast<EvalFunction*>(sourceC);
    if (fs)
      fs->setParam(name,value);
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
  explicit SIMDarcyCoSTA(Darcy& integrand,
                         const std::vector<unsigned char>& nf) :
    SIMDarcy<Dim>(integrand, nf)
  {
  }

  void setParam(const std::string& name, double value)
  {
    this->drc.setParam(name, value);
    if (this->mySol) {
      for (size_t i = 0; i < 2; ++i) {
        EvalFunction* f = dynamic_cast<EvalFunction*>(this->mySol->getScalarSol(i));
        if (f)
          f->setParam(name, value);
      }

      for (size_t i = 0; i < 2; ++i) {
        VecFuncExpr* v = dynamic_cast<VecFuncExpr*>(this->mySol->getScalarSecSol(0));
        if (v)
          v->setParam(name, value);
      }

      for (auto& it : this->myScalars) {
        EvalFunction* f = dynamic_cast<EvalFunction*>(it.second);
        if (f)
          f->setParam(name, value);
      }
      for (auto& it : this->myScalars) {
        VecFuncExpr* v = dynamic_cast<VecFuncExpr*>(it.second);
        if (v)
          v->setParam(name, value);
      }
    }
  }

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
    DarcyPreParse preparse(infile);

    std::vector<unsigned char> nf;
    if (preparse.twofield) {
      if (ASMmxBase::Type == ASMmxBase::FULL_CONT_RAISE_BASIS2 ||
          ASMmxBase::Type == ASMmxBase::REDUCED_CONT_RAISE_BASIS2)
        nf = {1,1};
      else
        nf = {2};
      integrand = std::make_unique<MixedDarcyCoSTA>(Dim::dimension, preparse.torder);
    } else {
      nf = {1};
      integrand = std::make_unique<DarcyCoSTA>(Dim::dimension, preparse.torder);
    }
    newModel = std::make_unique<SIMDarcyCoSTA<Dim>>(*integrand, nf);
    model = newModel.get();
    solModel = newModel.get();
    if (ConfigureSIM(static_cast<SIMDarcy<Dim>&>(*newModel),
                     const_cast<char*>(infile.c_str())))
      throw std::runtime_error("Error reading input file");
  }

  std::unique_ptr<Darcy> integrand; //!< Pointer to integrand instance
};


void export_Darcy(pybind11::module& m)
{
  CoSTAModule<SIMDarcyCoSTA>::pyExport(m, "Darcy");
}
