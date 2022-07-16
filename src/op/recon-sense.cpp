#include "recon-sense.hpp"

namespace rl {

ReconSENSE::ReconSENSE(GridBase<Cx> *gridder, Cx4 const &maps, SDCOp *sdc, bool const toe)
  : nufft_{LastN<3>(maps.dimensions()), gridder, sdc, toe}
  , sense_{maps, gridder->inputDimensions()[1]}
{
}

ReconSENSE::~ReconSENSE() {}

auto ReconSENSE::inputDimensions() const -> InputDims
{
  return sense_.inputDimensions();
}

auto ReconSENSE::outputDimensions() const -> OutputDims
{
  return nufft_.outputDimensions();
}

auto ReconSENSE::A(Input const &x) const -> Output
{
  Log::Debug("Starting ReconOp forward. Norm {}", Norm(x));
  auto const start = Log::Now();
  auto const y = nufft_.A(sense_.A(x));
  Log::Debug("Finished ReconOp forward. Norm {}. Took {}", Norm(y), Log::ToNow(start));
  return y;
}

auto ReconSENSE::Adj(Output const &x) const -> Input
{
  Log::Debug("Starting ReconOp adjoint. Norm {}", Norm(x));
  auto const start = Log::Now();
  Input y(inputDimensions());
  y.device(Threads::GlobalDevice()) = sense_.Adj(nufft_.Adj(x));
  Log::Debug("Finished ReconOp adjoint. Norm {}. Took {}.", Norm(y), Log::ToNow(start));
  return y;
}

auto ReconSENSE::AdjA(Input const &x) const -> Input
{
  Log::Debug("Starting ReconOp adjoint*forward. Norm {}", Norm(x));
  Input y(inputDimensions());
  auto const start = Log::Now();
  y.device(Threads::GlobalDevice()) = sense_.Adj(nufft_.AdjA(sense_.A(x)));
  Log::Debug("Finished ReconOp adjoint*forward. Norm {}. Took {}", Norm(y), Log::ToNow(start));
  return y;
}

} // namespace rl
