#pragma once

#include "nufft.hpp"
#include "recon.hpp"
#include "sdc.hpp"
#include "sense.hpp"

namespace rl {

struct ReconSENSE final : ReconOp
{
  using typename ReconOp::Input;
  using typename ReconOp::Output;

  ReconSENSE(GridBase<Cx> *gridder, Cx4 const &maps, SDCOp *sdc = nullptr, bool const toeplitz = false);
  ~ReconSENSE();

  InputDims inputDimensions() const;
  OutputDims outputDimensions() const;

  auto A(Input const &x) const -> Output;
  auto Adj(Output const &x) const -> Input;
  auto AdjA(Input const &x) const -> Input;

private:
  NUFFTOp nufft_;
  SenseOp sense_;
};

} // namespace rl
