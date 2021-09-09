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

template<template<class Dim> class Sim>
struct CoSTASIMAllocator
{
  template<class Dim>
   void allocate(std::unique_ptr<Sim<Dim>>& newModel, SIMbase*& model,
                 SIMsolution*& solModel, const std::string& infile);
};


template<template<class Dim> class Sim>
class CoSTAModule {
public:
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

   size_t ndof;

protected:
   void setDiscreteLoad(const std::vector<double>* load)
   {
     if (model1D)
       model1D->setDiscreteLoad(load);
     else if (model2D)
       model2D->setDiscreteLoad(load);
     else
       model3D->setDiscreteLoad(load);
   }

   bool solveStep(TimeStep& tp)
   {
     if (model1D)
       return model1D->solveStep(tp);
     else if (model2D)
       return model2D->solveStep(tp);
     else
       return model3D->solveStep(tp);
   }

   CoSTASIMAllocator<Sim> allocator;
   std::unique_ptr<Sim<SIM1D>> model1D;
   std::unique_ptr<Sim<SIM2D>> model2D;
   std::unique_ptr<Sim<SIM3D>> model3D;
   SIMsolution* solModel;
   SIMbase* model;
};

#endif
