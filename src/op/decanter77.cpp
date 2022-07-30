#include "decanter-internal.hpp"

namespace rl {

template std::unique_ptr<GridBase<Cx>>
make_decanter_internal<7, 7>(Kernel const *k, Mapping const &m, Cx4 const &kS);

template std::unique_ptr<GridBase<Cx>>
make_decanter_internal<7, 7>(Kernel const *k, Mapping const &m, Cx4 const &kS, R2 const &basis);

} // namespace rl
