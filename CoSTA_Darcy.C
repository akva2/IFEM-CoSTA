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
#include "DarcyTransport.h"
#include "SIMDarcy.h"

#include "AlgEqSystem.h"
#include "ElmMats.h"
#include "ElmNorm.h"
#include "FiniteElement.h"
#include "ForceIntegrator.h"
#include "TimeIntUtils.h"
#include "Utilities.h"

#include <tinyxml2.h>


/*!
  \brief Preparsing to check the Darcy formulation to use.
*/

class DarcyPreParse : public XMLInputBase
{
public:
  bool tracer = false; //!< True to include a tracer field
  int torder = 0; //!< Time integration order

  //! \brief Constructor.
  DarcyPreParse(const std::string& file)
  {
    this->readXML(file.c_str(), false);
  }

protected:
  //! \brief Parse an XML element.
  bool parse(const tinyxml2::XMLElement* elem)
  {
    if (!strcasecmp(elem->Value(),"timestepping")) {
      std::string type;
      if (utl::getAttribute(elem,"type",type))
        torder = TimeIntegration::Order(TimeIntegration::get(type));
    } else if (!strcasecmp(elem->Value(),"darcy"))
      utl::getAttribute(elem,"tracer",tracer);

    return true;
  }
};


/*!
 \brief Integrand for Darcy with CoSTA additions.
*/

class DarcyCoSTA : public Darcy
{
public:
  //! \brief The constructor forwards to the parent class constructor.
  //! \param[in] n Number of spatial dimensions
  //! \param[in] ord Order of time stepping scheme (BE/BDF2)
  DarcyCoSTA(unsigned short int n, int ord) : Darcy(n,ord) {}

  using Darcy::finalizeElement;
  //! \brief Finalizes the element quantities after the numerical integration.
  //! \details This method is invoked once for each element, after the numerical
  //! integration loop over interior points is finished and before the resulting
  //! element quantities are assembled into their system level equivalents.
  //! It is used here to calculate the linear residual, b := b - A*u,
  //! if requested. Note that CoSTAModule::residual() negates this to
  //! obtain the additive right-hand-side correction used by CoSTA.
  bool finalizeElement(LocalIntegral& elmInt) override
  {
    if (m_mode == SIM::RHS_ONLY) {
      ElmMats& A = static_cast<ElmMats&>(elmInt);
      A.A[0].multiply(A.vec[0], A.b[0], -1.0, 1.0);
    }

    return true;
  }

  //! \brief Set a parameter in the material and source functions.
  //! \param name Name of parameter
  //! \param value Value of parameter
  void setParam(const std::string& name, double value) override
  {
    if (mat)
      mat->setParam(name, value);
    if (source.get())
      source->setParam(name, value);
  }
};


/*!
 \brief Integrand for Darcy transport with CoSTA additions.
*/

class DarcyTransportCoSTA : public DarcyTransport
{
public:
  //! \brief The constructor forwards to the parent class constructor.
  //! \param[in] n Number of spatial dimensions
  //! \param[in] ord Order of time stepping scheme (BE/BDF2)
  DarcyTransportCoSTA(unsigned short int n, int ord) : DarcyTransport(n,ord) {}

  //! \brief Set a parameter in the material and source functions.
  //! \param name Name of parameter
  //! \param value Value of parameter
  void setParam(const std::string& name, double value) override
  {
    if (mat)
      mat->setParam(name, value);
    if (source.get())
      source->setParam(name, value);
    if (sourceC.get())
      sourceC->setParam(name, value);
  }
};


/*!
  \brief Class representing the integrand for computing the integral of the concentration.
*/

class DarcyConcentrationIntegral : public ForceBase
{
public:
  //! \brief Constructor for global force resultant integration.
  //! \param[in] dp The Darcy problem to evaluate concentration integral for
  explicit DarcyConcentrationIntegral(DarcyTransport& dp) : ForceBase(dp) {}

  using ForceBase::evalInt;
  //! \brief Evaluates the integrand at a boundary point.
  //! \param elmInt The local integral object to receive the contributions
  //! \param[in] fe Finite element data of current integration point
  bool evalInt(LocalIntegral& elmInt, const FiniteElement& fe,
               const TimeDomain&, const Vec3&) const override
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
  //! \param elmInt Local integral for element
  //!
  //! \details This method is invoked once before starting the numerical
  //! integration loop over the Gaussian quadrature points over an element.
  //! It is supposed to perform all the necessary internal initializations
  //! needed before the numerical integration is started for current element.
  bool initElement(const IntVec& MNPC,
                   const FiniteElement&, const Vec3&, size_t,
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
class SIMDarcyCoSTA : public SIMDarcy<Dim>, public CoSTASIMHelper
{
public:
  //! \brief Constructor.
  //! \param dp Reference to integrand to use
  //! \param nf Number of fields
  SIMDarcyCoSTA(Darcy& dp, unsigned char nf) : SIMDarcy<Dim>(dp,nf) {}

  //! \brief Set a parameter in the functions.
  //! \param name Name of parameter
  //! \param value Value of parameter
  void setParam(const std::string& name, double value)
  {
    Dim::myProblem->setParam(name, value);
    if (Dim::mySol) {
      for (size_t i = 0; i < 2; ++i)
        if (RealFunc* f = Dim::mySol->getScalarSol(i); f)
          f->setParam(name, value);

      for (size_t i = 0; i < 2; ++i)
        if (VecFunc* v = Dim::mySol->getScalarSecSol(0); v)
          v->setParam(name, value);

      for (auto& it : Dim::myScalars)
        if (it.second)
          it.second->setParam(name, value);

      for (auto& it : Dim::myVectors)
        if (it.second)
          it.second->setParam(name, value);
    }
  }

  //! \brief Returns analytical solutions projected on primary basis.
  //! \param t Time to evaluate at
  std::map<std::string,RealArray> getAnaSols(double t)
  {
    return this->CoSTASIMHelper::getAsolScalar(t, Dim::mySol, this);
  }

  //! \brief Returns a quantity of interest.
  //! \param[in] u Solution vector to evaluate for
  //! \param[in] time Parameters for time-dependent simulations
  //! \param[in] qi Name of the quantity of interest to return
  RealArray getQI(const RealArray& u,
                  const TimeDomain& time,
                  const std::string& qi)
  {
    RealArray integral;
    const auto it = myQI.find(qi);
    if (it != myQI.end()) {
      it->second.itg->initBuffer(this->getNoElms());
      SIM::integrate({u}, this, it->second.code, time, it->second.itg.get());
      it->second.itg->assemble(integral);
    }
    return integral;
  }

protected:
  //! \brief Parses a data section from an XML element.
  bool parse(const tinyxml2::XMLElement* elem) override
  {
    if (strcasecmp(elem->Value(),"darcy"))
      return this->Dim::parse(elem);

    const char* qoi = "quantities_of_interest";
    for (const tinyxml2::XMLElement* child2 = elem->FirstChildElement(qoi);
         child2; child2 = child2->NextSiblingElement(qoi))
      for (const tinyxml2::XMLElement* child = child2->FirstChildElement("qi");
           child; child = child->NextSiblingElement("qi")) {
        std::string name, set, type;
        if (utl::getAttribute(child,"name",name) && !name.empty())
          if (utl::getAttribute(child,"set",set) && !set.empty())
            if (utl::getAttribute(child,"type",type) &&
                type == "ConcentrationIntegral") {
              DarcyTransport& itg = static_cast<DarcyTransport&>(*Dim::myProblem);
              QI qi {
                std::make_unique<DarcyConcentrationIntegral>(itg),
                this->getUniquePropertyCode(set)
              };
              myQI.emplace(name, std::move(qi));
              IFEM::cout << "Quantity of interest: name = " << name
                         << " set = " << set << " type = " << type << std::endl;
            }
      }

    return this->SIMDarcy<Dim>::parse(elem);
  }

  //! \brief Assembles problem-dependent discrete terms, if any.
  bool assembleDiscreteTerms(const IntegrandBase*, const TimeDomain&) override
  {
    return this->assembleDiscreteLoad(this->getNoDOFs(), Dim::mySam,
                                      Dim::myEqSys->getVector(0));
  }

private:
  //! \brief Struct describing a quantity of interest.
  struct QI {
    std::unique_ptr<ForceBase> itg; //!< Integrand to use for evaluation
    int code = 0; //!< Property code
  };

  std::map<std::string, QI> myQI; //!< Map of quantities of interest
};


//! \brief Specialization for SIMDarcy.
template<>
struct CoSTASIMAllocator<SIMDarcyCoSTA> {
  //! \brief Allocates a Darcy simulator for given dimensionality.
  //! \param[out] newModel Allocated SIMDarcy instance
  //! \param[out] model Pointer to SIMbase interface for \a newModel
  //! \param[out] solModel Pointer to SIMsolution interface for \a newModel
  //! \param[in] infile Input file to parse model description from
  template<class Dim>
  void allocate(std::unique_ptr<SIMDarcyCoSTA<Dim>>& newModel, SIMbase*& model,
                SIMsolution*& solModel, const std::string& infile)
  {
    DarcyPreParse preparse(infile);

    if (preparse.tracer)
      integrand = std::make_unique<DarcyTransportCoSTA>(Dim::dimension, preparse.torder);
    else
      integrand = std::make_unique<DarcyCoSTA>(Dim::dimension, preparse.torder);
    newModel = std::make_unique<SIMDarcyCoSTA<Dim>>(*integrand, preparse.tracer ? 2 : 1);
    if (!newModel->read(infile.c_str()))
      throw std::runtime_error("Error reading input file");
    if (!newModel->preprocess())
      throw std::runtime_error("Error preprocessing the model");
    if (!newModel->init())
      throw std::runtime_error("Error initializing the model");
    model = newModel.get();
    solModel = newModel.get();
  }

  std::unique_ptr<Darcy> integrand; //!< Pointer to integrand instance
};


void export_Darcy(pybind11::module& m)
{
  CoSTAModule<SIMDarcyCoSTA>::pyExport(m, "Darcy");
}
