#pragma once

#include "operator.hpp"

#include "nufft.hpp"
#include "sdc.hpp"
#include "sense.hpp"

namespace rl {

struct ReconOp : Operator<4, 3>
{
  using typename Operator<4, 3>::Input;
  using typename Operator<4, 3>::Output;

  virtual ~ReconOp() {}

  virtual auto A(Input const &x) const -> Output = 0;
  virtual auto Adj(Output const &x) const -> Input = 0;
  virtual auto AdjA(Input const &x) const -> Input = 0;

};

} // namespace rl
