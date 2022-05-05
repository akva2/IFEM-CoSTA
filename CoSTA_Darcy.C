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
#include "DarcySolutions.h"
#include "MixedDarcy.h"
#include "SIMDarcy.h"

#include "AlgEqSystem.h"
#include "ASMmxBase.h"
#include "ElmMats.h"
#include "ElmNorm.h"
#include "ExprFunctions.h"
#include "ForceIntegrator.h"
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
  //! \param[in] torder Order of time stepping scheme (BE/BDF2)
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
    DiracSum* ds = dynamic_cast<DiracSum*>(source);
    if (ds)
      ds->setParam(name,value);
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
  //! \param[in] torder Order of time stepping scheme (BE/BDF2)
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
    DiracSum* ds = dynamic_cast<DiracSum*>(source);
    if (ds)
      ds->setParam(name,value);
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
  \brief Class representing the integrand for computing the integral of the concentration.
*/

class DarcyConcentrationIntegral : public ForceBase
{
public:
  //! \brief Constructor for global force resultant integration.
  //! \param[in] p The heat equation problem to evaluate fluxes for
  explicit DarcyConcentrationIntegral(MixedDarcy& p) : ForceBase(p) {}

  using ForceBase::evalInt;
  //! \brief Evaluates the integrand at a boundary point.
  //! \param elmInt The local integral object to receive the contributions
  //! \param[in] fe Finite element data of current integration point
  //! \param[in] time Parameters for nonlinear and time-dependent simulations
  //! \param[in] X Cartesian coordinates of current integration point
  bool evalInt(LocalIntegral& elmInt, const FiniteElement& fe,
               const TimeDomain& time, const Vec3& X) const override
  {
    ElmNorm& elmNorm = static_cast<ElmNorm&>(elmInt);

    double C = fe.N.dot(elmNorm.vec[1]);

    elmNorm[0] += C*fe.detJxW;

    return true;
  }

  //! \brief Returns the number of force components.
  size_t getNoComps() const override { return 1; }

  using ForceBase::initElement;
  //! \brief Initializes current element for numerical integration.
  //! \param[in] MNPC Matrix of nodal point correspondance for current element
  //! \param[in] fe Nodal and integration point data for current element
  //! \param[in] X0 Cartesian coordinates of the element center
  //! \param[in] nPt Number of integration points in this element
  //! \param elmInt Local integral for element
  //!
  //! \details This method is invoked once before starting the numerical
  //! integration loop over the Gaussian quadrature points over an element.
  //! It is supposed to perform all the necessary internal initializations
  //! needed before the numerical integration is started for current element.
  //! Reimplement this method for problems requiring the element center and/or
  //! the number of integration points during/before the integrand evaluations.
  bool initElement(const std::vector<int>& MNPC,
                   const FiniteElement& fe,
                   const Vec3& X0, size_t nPt,
                   LocalIntegral& elmInt) override
  {
    return myProblem.initElement(MNPC,elmInt);
   }

  //! \brief This is a volume integrand.
  bool hasInteriorTerms() const override { return true; }
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
  //! \param nf Number of fields on each basis
  explicit SIMDarcyCoSTA(Darcy& integrand,
                         const std::vector<unsigned char>& nf) :
    SIMDarcy<Dim>(integrand, nf)
  {
  }

  //! \brief Set a parameter in the functions.
  //! \param name Name of parameter
  //! \param value Value of parameter
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

  //! \brief Returns a quantity of interest.
  //! \param[in] u Solution vector to evaluate for
  //! \param[in] time Parameters for nonlinear and time-dependent simulations
  //! \param[in] qi Name of QI
  RealArray getQI(const RealArray& u,
                  const TimeDomain& time,
                  const std::string& qi)
  {
    Vector integral;
    const auto it = myQI.find(qi);
    if (it != myQI.end()) {
      Vectors solution(1);
      solution[0].resize(u.size());
      std::copy(u.begin(), u.end(), solution[0].begin());
      it->second.itg->initBuffer(this->getNoElms());
      SIM::integrate(solution, this, it->second.code, time, it->second.itg.get());
      it->second.itg->assemble(integral);
    }
    return integral;
  }

protected:
  //! \brief Parses a data section from an XML element.
  bool parse(const TiXmlElement* elem) override
  {
    if (!strcasecmp(elem->Value(),"darcy")) {
      const TiXmlElement* child2 = elem->FirstChildElement();
      for (; child2; child2 = child2->NextSiblingElement())
        if (!strcasecmp(child2->Value(), "quantities_of_interest"))  {
          const TiXmlElement* child = child2->FirstChildElement("qi");
          for (; child; child = child->NextSiblingElement("qi")) {
            std::string name, set, type;
            QI qi;
            utl::getAttribute(child, "name", name);
            utl::getAttribute(child, "set", set);
            utl::getAttribute(child, "type", type);
            if (type == "ConcentrationIntegral" && this->getProblem()->getNoFields() > 1) {
              MixedDarcy& itg = static_cast<MixedDarcy&>(*this->myProblem);
              qi.itg = std::make_unique<DarcyConcentrationIntegral>(itg);
              qi.code = this->getUniquePropertyCode(set);
            }
            if (!name.empty() && !set.empty() && qi.itg) {
              myQI.emplace(std::make_pair(name, std::move(qi)));
              IFEM::cout << "Quantity of interest: name = " << name
                         << " set = " << set << " type = " << type << std::endl;
            }
          }
        }
    }

    return this->SIMDarcy<Dim>::parse(elem);
  }

  //! \brief Assembles problem-dependent discrete terms, if any.
  bool assembleDiscreteTerms(const IntegrandBase*,
                             const TimeDomain&) override
  {
    return this->assembleDiscreteLoad(this->getNoDOFs(),
                                      this->mySam,
                                      this->myEqSys->getVector(0));
  }

  //! \brief Struct describing a quantity of interest.
  struct QI {
    std::unique_ptr<ForceBase> itg; //!< Integrand to use for evaluation
    int code; //!< Property code
  };

  std::map<std::string, QI> myQI; //!< Map of quantities of interest
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
