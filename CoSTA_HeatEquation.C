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

#include "AlgEqSystem.h"
#include "AnaSol.h"
#include "ASMstruct.h"
#include "BDF.h"
#include "ElmMats.h"
#include "EqualOrderOperators.h"
#include "ExprFunctions.h"
#include "Functions.h"
#include "IntegrandBase.h"
#include "SIMconfigure.h"
#include "SystemMatrix.h"
#include "Utilities.h"

#include <tinyxml.h>


/*!
 \brief Integrand for HeatEquation with CoSTA additions.
*/

class HeatEquationCoSTA : public IntegrandBase
{
public:
  using WeakOps = EqualOrderOperators::Weak; //!< Convenience rename

  //! \brief Constructor.
  //! \param n Number of spatial dimensions
  explicit HeatEquationCoSTA(unsigned short int n) :
    IntegrandBase(n), bdf(1), flux(nullptr)
  {
    primsol.resize(2);
    sourceTerm = nullptr;
  }

  //! \brief Defines the source term.
  void setSource(RealFunc* src) { sourceTerm.reset(src); }

  //! \brief Defines the diffusivity parameter.
  void setDiffusivity(RealFunc* d) { diffFunc.reset(d); }

  //! \brief Defines the flux function.
  void setFlux(RealFunc* f) { flux = f; }

  //! \brief Evaluates the source term (if any) at a specified point.
  double getSource(const Vec3& X) const
  {
    return sourceTerm ? (*sourceTerm)(X) : 0.0;
  }

  //! \brief Returns thermal diffusivity at a specified point.
  double getDiffusivity(const Vec3& X) const
  {
    return diffFunc ? (*diffFunc)(X) : 1.0;
  }

  //! \brief Evaluates the integrand at an interior point.
  //! \param elmInt The local integral object to receive the contributions
  //! \param[in] fe Finite element data of current integration point
  //! \param[in] time Parameters for nonlinear and time-dependent simulations
  //! \param[in] X Cartesian coordinates of current integration point
  bool evalInt(LocalIntegral& elmInt, const FiniteElement& fe,
               const TimeDomain& time, const Vec3& X) const override
  {
    Matrix& A = static_cast<ElmMats&>(elmInt).A.front();
    Vector& b = static_cast<ElmMats&>(elmInt).b.front();

    double kappa = this->getDiffusivity(X);
    double rhocp = 1.0;

    double theta = 0.0;
    for (int t = 1; t <= bdf.getOrder(); t++) {
      double val = fe.N.dot(elmInt.vec[t]);
      theta -= bdf[t]/time.dt*val;
    }

    WeakOps::Laplacian(A,fe,kappa);
    WeakOps::Mass(A,fe,rhocp*bdf[0]/time.dt);
    WeakOps::Source(b,fe,rhocp*theta+this->getSource(X));

    return true;
  }

  using IntegrandBase::evalBou;
  //! \brief Evaluates the integrand at a boundary point.
  //! \param elmInt The local integral object to receive the contributions
  //! \param[in] fe Finite element data of current integration point
  //! \param[in] X Cartesian coordinates of current integration point
  //! \param[in] normal Boundary normal vector at current integration point
  bool evalBou(LocalIntegral& elmInt,
               const FiniteElement& fe,
               const Vec3& X, const Vec3& normal) const override
  {
    if (!flux) {
      std::cerr <<" *** HeatEquationCoSTA::evalBou: No fluxes."<< std::endl;
      return false;
    }

    // Evaluate the Neumann value
    double T = (*flux)(X);
    double kappa = this->getDiffusivity(X);

    // Integrate the Neumann value
    WeakOps::Source(static_cast<ElmMats&>(elmInt).b.front(),fe,kappa*T);

    return true;
  }

  using IntegrandBase::finalizeElement;
  //! \brief Finalizes the element quantities after the numerical integration.
  //! \details This method is invoked once for each element, after the numerical
  //! integration loop over interior points is finished and before the resulting
  //! element quantities are assembled into their system level equivalents.
  //! It is used here to calculate the linear residual if requested.
  bool finalizeElement (LocalIntegral& elmInt) override
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
  void setParam(const std::string& name, double value)
  {
    EvalFunction* f = dynamic_cast<EvalFunction*>(sourceTerm.get());
    if (f)
      f->setParam(name, value);
    f = dynamic_cast<EvalFunction*>(flux);
    if (f)
      f->setParam(name, value);
    f = dynamic_cast<EvalFunction*>(diffFunc.get());
    if (f)
      f->setParam(name, value);
  }

protected:
  TimeIntegration::BDF bdf; //!< BDF helper class
  RealFunc* flux; //!< Flux function
  std::unique_ptr<RealFunc> sourceTerm;  //!< Pointer to source term
  std::unique_ptr<RealFunc> diffFunc;  //!< Pointer to diffusivity function
};


/*!
 \brief CoSTA simulator for HeatEquation.
*/

template<class Dim>
class SIMHeatCoSTA : public Dim,
                     public SIMsolution,
                     public CoSTASIMHelper
{
public:
  using SetupProps = bool; //!< Dummy setup properties

  //! \brief Constructor
  //! \param torder Time integration order
  SIMHeatCoSTA(int torder) : Dim(1), heq(Dim::dimension)
  {
    Dim::myProblem = &heq;
    Dim::myHeading = "Heat equation solver";
  }

  //! \brief The destructor clears up some pointers.
  virtual ~SIMHeatCoSTA()
  {
    Dim::myProblem = nullptr;
    Dim::myInts.clear();

    // To prevent the SIMbase destructor try to delete already deleted functions
    if (aCode[0] > 0) Dim::myScalars.erase(aCode[0]);
    if (aCode[1] > 0) Dim::myVectors.erase(aCode[1]);
  }

  using Dim::parse;
  //! \brief Parses a data section from an XML element.
  bool parse(const TiXmlElement* elem) override
  {
    if (strcasecmp(elem->Value(),"heatequation"))
      return this->Dim::parse(elem);

    const TiXmlElement* child = elem->FirstChildElement();
    for (; child; child = child->NextSiblingElement())
      if (!strcasecmp(child->Value(),"anasol")) {
        IFEM::cout <<"\tAnalytical solution: Expression"<< std::endl;
        if (!Dim::mySol)
          Dim::mySol = new AnaSol(child);

        // Define the analytical boundary traction field
        int code = 0;
        if (utl::getAttribute(child,"code",code))
          if (code > 0 && Dim::mySol->getScalarSecSol())
          {
            this->setPropertyType(code,Property::NEUMANN);
            Dim::myVectors[code] = Dim::mySol->getScalarSecSol();
          }
      }

      else if (!strcasecmp(child->Value(),"source")) {
        std::string type;
        utl::getAttribute(child, "type", type, true);
        if (type == "expression" && child->FirstChild()) {
          IFEM::cout <<"\tSource function (expression)";
          RealFunc* srcF = utl::parseRealFunc(child->FirstChild()->Value(),type);
          IFEM::cout << std::endl;
          heq.setSource(srcF);
        }
      }

      else if (!strcasecmp(child->Value(),"diffusivity")) {
        std::string type;
        utl::getAttribute(child, "type", type, true);
        if (type == "expression" && child->FirstChild()) {
          IFEM::cout <<"\tDiffusivity function (expression)";
          RealFunc* srcF = utl::parseRealFunc(child->FirstChild()->Value(),type);
          IFEM::cout << std::endl;
          heq.setDiffusivity(srcF);
        }
      }

      else
        this->Dim::parse(child);

    return true;
  }

  //! \brief Returns the name of this simulator (for use in the HDF5 export).
  std::string getName() const override { return "HeatEquation"; }

  //! \brief Initializes the temperature solution vectors.
  void initSol()
  {
    this->initSystem(Dim::opt.solver);

    size_t n, nSols = this->getNoSolutions();
    std::string str = "temperature1";
    this->initSolution(this->getNoDOFs(),nSols);
    for (n = 0; n < nSols; n++, str[11]++)
      this->registerField(str,solution[n]);
  }

  //! \brief Computes the solution for the current time step.
  bool solveStep(TimeStep& tp)
  {
    PROFILE1("SIMHeatCoSTA::solveStep");

    if (Dim::msgLevel >= 0)
      IFEM::cout <<"\n  step = "<< tp.step
                 <<"  time = "<< tp.time.t << std::endl;

    this->setQuadratureRule(Dim::opt.nGauss[0]);
    if (!this->assembleSystem(tp.time,solution))
      return false;

    if (!this->solveSystem(solution.front(),Dim::msgLevel-1,"temperature "))
      return false;

    if (Dim::msgLevel == 1)
    {
      size_t iMax[1];
      double dMax[1];
      double normL2 = this->solutionNorms(solution.front(),dMax,iMax,1);
      IFEM::cout <<"  Temperature summary: L2-norm         : "<< normL2
                 <<"\n                       Max temperature : "<< dMax[0]
                 << std::endl;
    }

    return true;
  }

  //! \brief Initializes for integration of Neumann terms for a given property.
  //! \param[in] propInd Physical property index
  bool initNeumann(size_t propInd) override
  {
    typename Dim::SclFuncMap::const_iterator tit = Dim::myScalars.find(propInd);
    if (tit == Dim::myScalars.end())
      return false;

    heq.setFlux(tit->second);
    return true;
  }

  //! \brief Performs some pre-processing tasks on the FE model.
  //! \details This method is reimplemented to couple the weak Dirichlet
  //! integrand to the Robin property codes.
  void preprocessA() override
  {
    Dim::myInts.insert(std::make_pair(0,Dim::myProblem));

    for (Property& p : Dim::myProps)
      if (p.pcode == Property::DIRICHLET_ANASOL) {
        if (!Dim::mySol->getScalarSol())
          p.pcode = Property::UNDEFINED;
        else if (aCode[0] == abs(p.pindx))
          p.pcode = Property::DIRICHLET_INHOM;
        else if (aCode[0] == 0)
        {
          aCode[0] = abs(p.pindx);
          Dim::myScalars[aCode[0]] = Dim::mySol->getScalarSol();
          p.pcode = Property::DIRICHLET_INHOM;
        }
        else
          p.pcode = Property::UNDEFINED;
      } else if (p.pcode == Property::NEUMANN_ANASOL) {
        if (!Dim::mySol->getScalarSecSol())
          p.pcode = Property::UNDEFINED;
        else if (aCode[1] == p.pindx)
          p.pcode = Property::NEUMANN;
        else if (aCode[1] == 0)
        {
          aCode[1] = p.pindx;
          Dim::myVectors[aCode[1]] = Dim::mySol->getScalarSecSol();
          p.pcode = Property::NEUMANN;
        }
        else
          p.pcode = Property::UNDEFINED;
      }
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
      for (auto& it : this->myScalars) {
        EvalFunction* f = dynamic_cast<EvalFunction*>(it.second);
        if (f)
          f->setParam(name, value);
      }
      for (auto& it : this->myScalars) {
        VecFuncExpr* f = dynamic_cast<VecFuncExpr*>(it.second);
        if (f)
          f->setParam(name, value);
      }
    }
    if (ICf.empty() && !this->myICs.empty())
      for (const auto& it : this->myICs)
        if (it.first == "nofile")
          for (const auto& ic : it.second) {
            RealFunc* f = utl::parseRealFunc(ic.function,ic.file_field,false);
            ICf[ic.sim_field] = std::unique_ptr<RealFunc>(f);
          }
    for (auto& it : ICf) {
      EvalFunction* f = dynamic_cast<EvalFunction*>(it.second.get());
      if (f)
        f->setParam(name, value);
    }
  }

  //! \brief Sets initial conditions.
  //! \details This has been overridden here so that the functions are not local
  //!          but live during the entire lifetime of the class. This in order to handle
  //!          parameter dependencies.
  void setInitialConditions()
  {
    for (const auto& it : this->myICs)
      if (it.first == "nofile")
        for (const auto& ic : it.second)
        {
          // Do we have this field?
          Vector* field = this->getField(ic.sim_field);
          if (!field) continue;
          this->project(*field,ICf[ic.sim_field].get(),ic.basis,ic.component-1,
                        this->getNoFields(ic.basis));
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

  std::map<std::string,std::unique_ptr<RealFunc>> ICf; //!< Initial condition functions
  HeatEquationCoSTA heq;  //!< Main integrand
  std::array<int,2> aCode = {0,0}; //!< Codes for analytic boundary conditions
};


//! \brief Partial specialization for configurator.
template<class Dim>
struct SolverConfigurator<SIMHeatCoSTA<Dim>> {
  //! \brief Setup a heat equation simulator.
  //! \param sim The simulator to set up
  //! \param[in] props Setup properties
  //! \param[in] infile The input file to parse
  int setup(SIMHeatCoSTA<Dim>& sim,
            const typename SIMHeatCoSTA<Dim>::SetupProps& props, char* infile)
  {
    // Read the input file
    ASMstruct::resetNumbering();
    if (!sim.readModel(infile))
      return 2;

    // Preprocess the model and establish FE data structures
    if (!sim.preprocess())
      return 3;

    // Initialize the linear equation system solver
    sim.initSol();

    return 0;
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
    if (ConfigureSIM(*newModel, const_cast<char*>(infile.c_str())))
      throw std::runtime_error("Error reading input file");
  }
};


void export_HeatEquation(pybind11::module& m)
{
  CoSTAModule<SIMHeatCoSTA>::pyExport(m, "HeatEquation");
}
