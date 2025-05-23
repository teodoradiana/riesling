#pragma once

#include "log.hpp"
#include "parameter.hpp"
#include "sequence.hpp"
#include "types.hpp"

namespace rl {

struct IR final : Sequence
{
  static const Index nParameters = 2;
  IR(Settings const &s);

  auto length() const -> Index;
  auto simulate(Eigen::ArrayXf const &p) const -> Cx2;
};

} // namespace rl
