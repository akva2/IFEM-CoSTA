// $Id$
//==============================================================================
//!
//! \file CoSTAModule.h
//!
//! \date Sep 9 2021
//!
//! \author Arne Morten Kvarving / SINTEF
//!
//! \brief Class adding a CoSTA interface to a simulator.
//!
//==============================================================================

#ifndef COSTA_MODULE_H_
#define COSTA_MODULE_H_

#include "AnaSol.h"
#include "Functions.h"
#include "IFEM.h"
#include "Profiler.h"
#include "SAM.h"
#include "SIM1D.h"
#include "SIM2D.h"
#include "SIM3D.h"
#include "SIMargsBase.h"
#include "SIMenums.h"
#include "SIMsolution.h"
#include "SystemMatrix.h"
#include "TimeStep.h"

#include <memory>
#include <string>
#include <variant>


/*!
 \brief Helper structure to allocate the simulator in a CoSTAModule
*/

template<template<class Dim> class Sim>
struct CoSTASIMAllocator
{
  //! \brief Method to allocate a given dimensionality of a simulator.
  //! \param newModel Simulator to allocate
  //! \param model Pointer to SIMbase interface for simulator
  //! \param solModel Pointer to SIMsolution interface for simulatorA
  //! \param infile Input file to parse.
  //!
  //! \details This struct must be specialized for a given simulator
  template<class Dim>
  void allocate(std::unique_ptr<Sim<Dim>>& newModel, SIMbase*& model,
                SIMsolution*& solModel, const std::string& infile);
};


/*!
 \brief Helper class to add CoSTA needs to a simulator.
*/

class CoSTASIMHelper
{
public:
  using AnasolMap = std::map<std::string,RealArray>; //!< Convenience type alias

  //! \brief Set an additional discrete load.
  //! \param vec Load to use (nullptr for none)
  void setDiscreteLoad(const RealArray* vec) { discreteLoad = vec;  }

protected:
  //! \brief Assembles the discrete load.
  //! \param nDofs Number of degrees of freedom
  //! \param sam The SAM object to use
  //! \param v The vector to assemble to
  bool assembleDiscreteLoad(size_t nDofs, const SAM* sam, SystemVector* v)
  {
    if (!discreteLoad)
      return true;

    if (discreteLoad->size() != nDofs)
      return false;

    for (size_t i = 1; i <= nDofs; ++i) {
      int eq = sam->getEquation(i, 1);
      if (eq != 0)
        v->getPtr()[eq-1] += (*discreteLoad)[i-1];
    }

    return true;
  }

  //! \brief Returns analytical solutions projected on primary basis (scalar equation).
  //! \param t Time to evaluate at
  //! \param anaSol Analytical solution to evaluate
  //! \param model Model to evaluate for
  static AnasolMap getAsolScalar(double t, const AnaSol* anaSol,
                                 const SIMbase* model)
  {
    std::map<std::string,RealArray> result;

    if (anaSol && anaSol->getScalarSol()) {
      Vector prim(model->getNoDOFs());
      model->project(prim, anaSol->getScalarSol(), 1, 0, 1,
                     SIMoptions::GLOBAL, t);
      result.insert(std::make_pair("primary",prim));
    }

    if (anaSol && anaSol->getScalarSecSol()) {
      Matrix ssol(model->getNoSpaceDim(), model->getNoDOFs());
      Vector secv(model->getNoSpaceDim()*model->getNoDOFs());
      model->project(secv, anaSol->getScalarSecSol(), 1, 0,
                     model->getNoSpaceDim(), SIMoptions::GLOBAL, t);
      ssol = secv;
      std::string sec = "secondary_x";
      for (size_t i = 1; i <= model->getNoSpaceDim(); ++i, ++sec.back())
        result.insert(std::make_pair(sec, ssol.getRow(i)));
    }

    return result;
  }

  const RealArray* discreteLoad = nullptr; //!< Additional discrete load vector
};


/*!
 \brief Class adding a CoSTA interface to a simulator.
*/

template<template<class Dim> class Sim>
class CoSTAModule {
public:
  using ParameterMap = std::map<std::string, std::variant<double,RealArray>>; //!< Map for parameters

  //! \brief Constructor.
  //! \param infile Input file to parse
  //! \param verbose If false, suppress stdout output
  CoSTAModule(const std::string& infile, bool verbose)
  {
    IFEM::Init(0,nullptr,"");
    if (!verbose)
      IFEM::cout.setPIDs(-1, 0);

    SIMargsBase preparser("dummy");
    if (!preparser.readXML(infile.c_str(), false))
      throw std::runtime_error("Error preparsing input file");

    if (preparser.dim == 1)
      allocator.allocate(model1D, model, solModel, infile);
    else if (preparser.dim == 2)
      allocator.allocate(model2D, model, solModel, infile);
    else
      allocator.allocate(model3D, model, solModel, infile);

    ndof = model->getNoDOFs();

    if (!verbose)
      model->getProcessAdm().cout.setPIDs(-1, 0);
  }

  //! \brief Perform a prediction step in a CoSTA loop.
  //! \param mu Model parameters
  //! \param uprev State to make a time-step from
  RealArray predict(const ParameterMap& mu, const RealArray& uprev)
  {
    TimeStep tp;
    tp.step = 1;
    std::tie(tp.time.dt, tp.time.t) = this->getTimeParams(mu);
    tp.starTime = 0.0;
    tp.stopTime = tp.time.t + tp.time.dt;

    solModel->setSolution(uprev,1);
    model->setMode(SIM::DYNAMIC);
    this->setDiscreteLoad(nullptr);
    this->setParameters(mu);

    Vector dummy;
    model->updateDirichlet(tp.time.t, &dummy);
    if (!this->solveStep(tp))
      throw std::runtime_error("Failure during prediction step");

    return solModel->getSolution(0);
  }

  //! \brief Perform a residual calculation step in a CoSTA loop.
  //! \param mu Model parameters
  //! \param uprev State to make a time-step from
  //! \param unext State to calculate residual for
  RealArray residual(const ParameterMap& mu,
                     const RealArray& uprev, const RealArray& unext)
  {
    TimeDomain time;
    std::tie(time.dt, time.t) = this->getTimeParams(mu);

    solModel->setSolution(uprev,1);
    solModel->setSolution(unext,0);
    model->setMode(SIM::RHS_ONLY);
    this->setDiscreteLoad(nullptr);
    this->setParameters(mu);
    model->updateDirichlet(time.t, nullptr);

    if (!model->assembleSystem(time, solModel->getSolutions()))
      throw std::runtime_error("Failure during residual calculation");

    Vector loadVec;
    model->extractLoadVec(loadVec);
    return std::move(loadVec);
  }

  //! \brief Perform a correction step in a CoSTA loop.
  //! \param mu Model parameters
  //! \param uprev State to make a time-step from
  //! \param sigma Right-hand-side correction to use
  RealArray correct(const ParameterMap& mu,
                    const RealArray& uprev, const RealArray& sigma)
  {
    TimeStep tp;
    tp.step = 1;
    std::tie(tp.time.dt, tp.time.t) = this->getTimeParams(mu);
    tp.starTime = 0.0;
    tp.stopTime = tp.time.t + tp.time.dt;

    solModel->setSolution(uprev,1);
    model->setMode(SIM::DYNAMIC);
    this->setDiscreteLoad(&sigma);
    this->setParameters(mu);

    Vector dummy;
    model->updateDirichlet(tp.time.t,&dummy);
    if (!this->solveStep(tp))
      throw std::runtime_error("Failure during correction step");

    return solModel->getSolution(0);
  }

  //! \brief Get the IDs of all Dirichlet DoFs.
  std::vector<int> dirichletDofs()
  {
    size_t nNodes = model->getNoNodes();
    std::vector<int> eqns, ret;

    for (size_t inode = 1; inode <= nNodes; inode++) {
      model->getSAM()->getNodeEqns(eqns, inode);
      auto first_dof = model->getSAM()->getNodeDOFs(inode).first;
      for (size_t i = 0; i < eqns.size(); i++) {
        if (eqns[i] <= 0)
          ret.push_back(first_dof + i);
      }
    }

    return ret;
  }

  //! \brief Returns initial condition for solution.
  RealArray initialCondition(const ParameterMap& mu)
  {
    this->setParameters(mu);
    if (model1D)
      model1D->setInitialConditions();
    else if (model2D)
      model2D->setInitialConditions();
    else
      model3D->setInitialConditions();

    return solModel->getSolution(0);
  }

  //! \brief Returns the analytical solution.
  //! \param mu Map of parameters
  std::map<std::string,RealArray> anaSol(const ParameterMap& mu)
  {
    double dt, t;
    std::tie(dt, t) = this->getTimeParams(mu);
    this->setParameters(mu);

    if (model1D)
      return model1D->getAnaSols(t);
    else if (model2D)
      return model2D->getAnaSols(t);
    else
      return model3D->getAnaSols(t);
  }

  size_t ndof; //!< Number of degrees of freedom in simulator

  //! \brief Static helper to export to python.
  //! \param m Module to export to
  //! \param name Name of python class
  static void pyExport(pybind11::module& m, const char* name)
  {
    pybind11::class_<CoSTAModule<Sim>>(m, name)
        .def(pybind11::init<const std::string&, bool>(),
             pybind11::arg("infile"),
             pybind11::arg("verbose") = true)
        .def("correct", &CoSTAModule<Sim>::correct)
        .def("predict", &CoSTAModule<Sim>::predict)
        .def("residual", &CoSTAModule<Sim>::residual)
        .def("dirichlet_dofs", &CoSTAModule<Sim>::dirichletDofs)
        .def("initial_condition", &CoSTAModule<Sim>::initialCondition)
        .def("anasol", &CoSTAModule<Sim>::anaSol)
        .def_readonly("ndof", &CoSTAModule<Sim>::ndof);
  }

protected:
  //! \brief Get a scalar parameter from map.
  //! \param map Map with parameters
  //! \param key Name of parameter
  double getScalarParameter(const ParameterMap& map, const std::string& key) const
  {
    const auto it = map.find(key);
    if (it == map.end())
      throw std::runtime_error("Need "+key+" in parameters");
    if (!std::holds_alternative<double>(it->second))
      throw std::runtime_error(key+" needs to be a double");
    return std::get<double>(it->second);
  }

  //! \brief Helper function to set a parameter in simulator.
  //! \param name Name of parameter
  //! \param value Value of parameter
  void setParam(const std::string& name, double value)
  {
    if (model1D)
      model1D->setParam(name, value);
    else if (model2D)
      model2D->setParam(name, value);
    else
      model3D->setParam(name, value);
  }

  //! \brief Set parameters from map in simulator.
  //! \param map Map of parameters
  void setParameters(const ParameterMap& map)
  {
    static const std::vector<std::string> blackList = {"dt", "t"};
    for (const auto& entry : map)
      if (std::holds_alternative<double>(entry.second))
        if (std::find(blackList.begin(), blackList.end(), entry.first) == blackList.end())
          this->setParam(entry.first, std::get<double>(entry.second));
  }

  //! \brief Helper function to provide RHS correction to simulator.
  //! \param load The RHS correction to provide to simulator
  void setDiscreteLoad(const RealArray* load)
  {
    if (model1D)
      model1D->setDiscreteLoad(load);
    else if (model2D)
      model2D->setDiscreteLoad(load);
    else
      model3D->setDiscreteLoad(load);
  }

  //! \brief Helper function to perform a time step.
  //! \param tp Time stepping parameters
  bool solveStep(const TimeStep& tp)
  {
    if (model1D)
      return model1D->solveStep(tp);
    else if (model2D)
      return model2D->solveStep(tp);
    else
      return model3D->solveStep(tp);
  }

  //! \brief Extract time stepping parameters from parameter map.
  std::tuple<double, double> getTimeParams(const ParameterMap& map) const
  {
    double dt = this->getScalarParameter(map, "dt");
    double t = dt;
    if (map.find("t") != map.end())
      t = this->getScalarParameter(map, "t");

    return std::make_tuple(dt,t);
  }

  CoSTASIMAllocator<Sim> allocator; //!< Helper for simulator allocation
  std::unique_ptr<Sim<SIM1D>> model1D; //!< Pointer to 1D instance of simulator
  std::unique_ptr<Sim<SIM2D>> model2D; //!< Pointer to 2D instance of simulator
  std::unique_ptr<Sim<SIM3D>> model3D; //!< Pointer to 3D instance of simulator
  SIMsolution* solModel; //!< Pointer to SIMsolution interface of simulator
  SIMbase* model; //!< Pointer to SIMbase interface of simulator
};

#endif
