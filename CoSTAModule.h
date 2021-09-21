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

#include "ASMbase.h"
#include "ElementBlock.h"
#include "IFEM.h"
#include "IntegrandBase.h"
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

#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>


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
  //!\ details This must be specialized for a given simulator
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
  //! \brief Set an additional discrete load.
  //! \param vec Load to use (nullptr for none)
  void setDiscreteLoad(const std::vector<double>* vec)
  {
    discreteLoad = vec;
  }

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


  const std::vector<double>* discreteLoad = nullptr; //!< Additional discrete load vector
};


/*!
 \brief Class adding a CoSTA interface to a simulator.
*/

template<template<class Dim> class Sim>
class CoSTAModule {
public:
  using ParameterMap = std::map<std::string, std::variant<double,std::vector<double>>>; //!< Map for parameters
  using NodeVec = std::vector<std::array<double,3>>; //!< Vector of nodal coordinates
  using ElemIndices = std::vector<std::vector<int>>; //!< Vector with elemental node indices
  using ElemBlock = std::pair<NodeVec,ElemIndices>; //!< A block of the grid

  //! \brief Constructor.
  //! \param infile Input file to parse
  CoSTAModule(const std::string& infile)
  {
    IFEM::Init(0,nullptr,"");
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
  }

  //! \brief Perform a prediction step in a CoSTA loop.
  //! \param mu Model parameters
  //! \param uprev State to make a time-step from
  std::vector<double> predict(const ParameterMap& mu,
                              const std::vector<double>& uprev)
  {
    TimeStep tp;
    tp.step = 1;
    std::tie(tp.time.dt, tp.time.t) = this->getTimeParams(mu);
    tp.starTime = 0.0;
    tp.stopTime = tp.time.t + tp.time.dt;

    Vector up;
    up = uprev;
    solModel->setSolution(up, 1);
    model->setMode(SIM::DYNAMIC);
    this->setDiscreteLoad(nullptr);

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
  std::vector<double> residual(const ParameterMap& mu,
                               const std::vector<double>& uprev,
                               const std::vector<double>& unext)
  {
    TimeDomain time;
    std::tie(time.dt, time.t) = this->getTimeParams(mu);

    Vector up;
    up = uprev;
    solModel->setSolution(up, 1);
    up = unext;
    solModel->setSolution(up, 0);
    model->setMode(SIM::RHS_ONLY);
    this->setDiscreteLoad(nullptr);
    model->updateDirichlet(time.t, nullptr);

    if (!model->assembleSystem(time, solModel->getSolutions()))
      throw std::runtime_error("Failure during residual calculation");

    model->extractLoadVec(up);

    return std::move(up);
  }

  //! \brief Perform a correction step in a CoSTA loop.
  //! \param mu Model parameters
  //! \param uprev State to make a time-step from
  //! \param sigma Right-hand-side correction to use
  std::vector<double> correct(const ParameterMap& mu,
                              const std::vector<double>& uprev,
                              const std::vector<double>& sigma)
  {
    TimeStep tp;
    tp.step = 1;
    std::tie(tp.time.dt, tp.time.t) = this->getTimeParams(mu);
    tp.starTime = 0.0;
    tp.stopTime = tp.time.t + tp.time.dt;

    Vector up;
    up = uprev;
    solModel->setSolution(up, 1);
    model->setMode(SIM::DYNAMIC);
    this->setDiscreteLoad(&sigma);

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

  //! \brief Returns a FE model for visualization.
  std::vector<ElemBlock> visualizationModel()
  {
    std::vector<ElemBlock> result;
    result.reserve(model->getNoPatches());
    std::vector<int> npe(3);
    npe[0] = npe[1] = npe[2] = 2;
    for (const ASMbase* pch : model->getFEModel()) {
      ElementBlock block(model->getNoSpaceDim() == 3 ? 8 : (model->getNoSpaceDim() == 2 ? 4 : 2));
      pch->tesselate(block, npe.data());
      NodeVec coords(block.getNoNodes());
      size_t idx = idx;
      for (size_t idx = 0; idx < block.getNoNodes(); ++idx) {
        coords[idx][0] = block.getCoord(idx).x;
        coords[idx][1] = block.getCoord(idx).y;
        coords[idx][2] = block.getCoord(idx).z;
      }
      ElemIndices elms(block.getNoElms());
      for (size_t idx = 0; idx < block.getNoElms(); ++idx) {
        std::copy(block.getElements()+idx*block.getNoElmNodes(),
                  block.getElements()+(idx+1)*block.getNoElmNodes(),
                  std::back_inserter(elms[idx]));
      }
      result.push_back(std::make_pair(coords, elms));
    }
    return result;
  }

  using ResultMap = std::map<std::string, std::vector<std::vector<double>>>; //!< Map with visualization results

  //! \brief Returns visualization results.
  //! \param primaries Primary solution vector
  ResultMap visualizationResults(const std::vector<double>& primaries)
  {
    Vector psol(primaries.size());
    for (size_t i = 0; i < primaries.size(); ++i)
      psol[i] = primaries[i];
    std::vector<int> npe(3);
    npe[0] = npe[1] = npe[2] = 2;
    ResultMap result;
    for (const ASMbase* pch : model->getFEModel()) {
      Vector lovec;
      pch->extractNodeVec(psol, lovec, 0);
      Matrix field;
      pch->evalSolution(field, lovec, npe.data());
      for (size_t i = 1; i <= field.rows(); ++i)
        result[model->getProblem()->getField1Name(i-1)].push_back(field.getRow(i));

      pch->evalSolution(field, *model->getProblem(), npe.data());
      for (size_t i = 1; i <= field.rows(); ++i)
        result[model->getProblem()->getField2Name(i-1)].push_back(field.getRow(i));
    }

    return result;
  }

  size_t ndof; //!< Number of degrees of freedom in simulator

  //! \brief Static helper to export to python.
  //! \param m Module to export to
  //! \param name Name of python class
  static void pyExport(pybind11::module& m, const char* name)
  {
    pybind11::class_<CoSTAModule<Sim>>(m, name)
        .def(pybind11::init<const std::string&>())
        .def("correct", &CoSTAModule<Sim>::correct)
        .def("predict", &CoSTAModule<Sim>::predict)
        .def("residual", &CoSTAModule<Sim>::residual)
        .def("dirichlet_dofs", &CoSTAModule<Sim>::dirichletDofs)
        .def("visualization_model", &CoSTAModule<Sim>::visualizationModel)
        .def("visualization_results", &CoSTAModule<Sim>::visualizationResults)
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

  //! \brief Helper function to provide RHS correction to simulator.
  //! \param load The RHS correction to provide to simulator
  void setDiscreteLoad(const std::vector<double>* load)
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
  bool solveStep(TimeStep& tp)
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
