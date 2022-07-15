#pragma once

#include "recon.hpp"

#include "nufft.hpp"
#include "sdc.hpp"
#include "sense.hpp"

namespace rl {

struct ReconRSS final : ReconOp
{
  using typename ReconOp::Input;
  using typename ReconOp::Output;

  ReconRSS(GridBase<Cx>*gridder, Sz3 const &dims, SDCOp *sdc = nullptr)
    : nufft_{dims, gridder, sdc}
  {
  }

  auto inputDimensions() const -> InputDims
  {
    return LastN<4>(nufft_.inputDimensions());
  }

  auto outputDimensions() const -> OutputDims
  {
    return nufft_.outputDimensions();
  }

  auto Adj(Output const &x) const -> Input
  {
    Log::Debug("Starting ReconRSSOp adjoint. Norm {}", Norm(x));
    auto const start = Log::Now();
    Cx5 const channels = nufft_.Adj(x);
    Cx4 y(inputDimensions());
    y.device(Threads::GlobalDevice()) = ConjugateSum(channels, channels).sqrt();
    Log::Debug("Finished ReconOp adjoint. Norm {}. Took {}", Norm(y), Log::ToNow(start));
    return y;
  }

  auto A(Input const &) const -> Output {
    Log::Fail("ReconRSS does not support Forward operation");
  }

  auto AdjA(Input const &) const -> Input {
    Log::Fail("ReconRSS does not support Adjoint*Forward operation");
  }

private:
  NUFFTOp nufft_;
};
}
