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

#include "IFEM.h"
#include "Profiler.h"
#include "SIM1D.h"
#include "SIM2D.h"
#include "SIM3D.h"
#include "SIMargsBase.h"
#include "SIMenums.h"
#include "SIMsolution.h"
#include "TimeStep.h"

#include <memory>
#include <string>


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
 \brief Class adding a CoSTA interface to a simulator.
*/

template<template<class Dim> class Sim>
class CoSTAModule {
public:
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
  std::vector<double> predict(const std::vector<double>& mu,
                              const std::vector<double>& uprev)
  {
    TimeStep tp;
    tp.step = 1;
    tp.time.t = tp.time.dt = mu[0];
    tp.starTime = 0.0;
    tp.stopTime = 2.0*mu[0];
    Vector up;
    up = uprev;
    solModel->setSolution(up, 1);
    model->setMode(SIM::DYNAMIC);
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
  std::vector<double> residual(const std::vector<double>& mu,
                               const std::vector<double>& uprev,
                               const std::vector<double>& unext)
  {
    TimeDomain time;
    time.t = time.dt = mu[0];
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
  std::vector<double> correct(const std::vector<double>& mu,
                              const std::vector<double>& uprev,
                              const std::vector<double>& sigma)
  {
    TimeStep tp;
    tp.step = 1;
    tp.time.t = tp.time.dt = mu[0];
    tp.starTime = 0.0;
    tp.stopTime = 2.0*mu[0];
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

  size_t ndof; //!< Number of degrees of freedom in simulator

protected:
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

   CoSTASIMAllocator<Sim> allocator; //!< Helper for simulator allocation
   std::unique_ptr<Sim<SIM1D>> model1D; //!< Pointer to 1D instance of simulator
   std::unique_ptr<Sim<SIM2D>> model2D; //!< Pointer to 2D instance of simulator
   std::unique_ptr<Sim<SIM3D>> model3D; //!< Pointer to 3D instance of simulator
   SIMsolution* solModel; //!< Pointer to SIMsolution interface of simulator
   SIMbase* model; //!< Pointer to SIMbase interface of simulator
};

#endif
