#pragma once

#include "decanter.hpp"

namespace rl {

template <int IP, int TP>
std::unique_ptr<GridBase<Cx>> make_decanter_internal(Kernel const *k, Mapping const &m, Cx4 const &kS)
{
  return std::make_unique<Decanter<IP, TP>>(dynamic_cast<SizedKernel<IP, TP> const *>(k), m, kS);
}

template <int IP, int TP>
std::unique_ptr<GridBase<Cx>> make_decanter_internal(Kernel const *k, Mapping const &m, Cx4 const &kS, R2 const &basis)
{
  return std::make_unique<Decanter<IP, TP>>(dynamic_cast<SizedKernel<IP, TP> const *>(k), m, kS, basis);
}

} // namespace rl
