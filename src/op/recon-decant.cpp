#include "recon-decant.hpp"

namespace rl {

ReconDecant::ReconDecant(GridBase<Cx> *gridder, SDCOp *sdc)
  : nufft_{LastN<3>(gridder->inputDimensions()), gridder, sdc, false}
{
}

ReconDecant::~ReconDecant() {}

auto ReconDecant::inputDimensions() const -> InputDims
{
  return LastN<4>(nufft_.inputDimensions());
}

auto ReconDecant::outputDimensions() const -> OutputDims
{
  return nufft_.outputDimensions();
}

auto ReconDecant::A(Input const &x) const -> Output
{
  Log::Debug("Starting Decant forward. Norm {}", Norm(x));
  auto const start = Log::Now();
  Cx5 temp(AddFront(inputDimensions(), 1));
  temp.chip(0,0) = x;
  auto const y = nufft_.A(temp);
  Log::Debug("Finished Decant forward. Norm {}. Took {}", Norm(y), Log::ToNow(start));
  return y;
}

auto ReconDecant::Adj(Output const &x) const -> Input
{
  Log::Debug("Starting Decant adjoint. Norm {}", Norm(x));
  auto const start = Log::Now();
  Input y(inputDimensions());
  y.device(Threads::GlobalDevice()) = nufft_.Adj(x).chip(0, 0);
  Log::Debug("Finished Decant adjoint. Norm {}. Took {}.", Norm(y), Log::ToNow(start));
  return y;
}

auto ReconDecant::AdjA(Input const &x) const -> Input
{
  Log::Fail("Not supported yet");
}

} // namespace rl
