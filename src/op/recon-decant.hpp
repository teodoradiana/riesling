#pragma once

#include "nufft.hpp"
#include "recon.hpp"
#include "sdc.hpp"
#include "sense.hpp"

namespace rl {

struct ReconDecant final : ReconOp
{
  using typename ReconOp::Input;
  using typename ReconOp::Output;

  ReconDecant(GridBase<Cx> *gridder, SDCOp *sdc = nullptr);
  ~ReconDecant();

  InputDims inputDimensions() const;
  OutputDims outputDimensions() const;

  auto A(Input const &x) const -> Output;
  auto Adj(Output const &x) const -> Input;
  auto AdjA(Input const &x) const -> Input;

private:
  NUFFTOp nufft_;
};

} // namespace rl
