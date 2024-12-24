#include "grad.hpp"

#include "../log.hpp"
#include "../sys/threads.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

namespace {
template <bool fwd, typename T1, typename T2, typename SzT>
inline auto ForwardDiff(T1 const &a, T2 &&b, SzT const dims, Index const dim)
{
  auto         sz = dims;
  decltype(sz) fm1, f0, fp1;
  fp1[dim] = 1;
  fm1[dim] = 0;
  sz[dim] -= 1;
  if constexpr (fwd) {
    f0[dim] = 0;
    b.slice(f0, sz).device(Threads::TensorDevice()) += (a.slice(fp1, sz) - a.slice(fm1, sz));
  } else {
    f0[dim] = 1;
    b.slice(f0, sz).device(Threads::TensorDevice()) += (a.slice(fm1, sz) - a.slice(fp1, sz));
  }
}

template <bool fwd, typename T1, typename T2, typename SzT>
inline auto CentralDiff0(T1 const &a, T2 &&b, SzT const dims, Index const dim)
{
  auto         sz = dims;
  decltype(sz) fm1, f0, fp1;
  fp1[dim] = 2;
  fm1[dim] = 0;
  f0[dim] = 1;
  sz[dim] -= 2;
  if constexpr (fwd) {
    b.slice(f0, sz).device(Threads::TensorDevice()) += (a.slice(fp1, sz) - a.slice(fm1, sz)) / b.slice(f0, sz).constant(2.f);
  } else {
    b.slice(f0, sz).device(Threads::TensorDevice()) += (a.slice(fm1, sz) - a.slice(fp1, sz)) / b.slice(f0, sz).constant(2.f);
  }
}

template <bool fwd, typename T1, typename T2, typename SzT>
inline auto CentralDiff1(T1 const &a, T2 &&b, SzT const dims, Index const dim)
{ // Thanks http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/central-differences/
  auto         sz = dims;
  decltype(sz) f0, fp1, fp2, fm1, fm2;
  f0[dim] = 2;
  fp1[dim] = 3;
  fp2[dim] = 4;
  fm1[dim] = 1;
  fm2[dim] = 0;
  sz[dim] -= 4;
  if constexpr (fwd) {
    b.slice(f0, sz).device(Threads::TensorDevice()) +=
      (a.slice(fm2, sz) - 8.f * a.slice(fm1, sz) + 8.f * a.slice(fp1, sz) - a.slice(fp2, sz)) / b.slice(f0, sz).constant(12.f);
  } else {
    b.slice(f0, sz).device(Threads::TensorDevice()) +=
      (a.slice(fp2, sz) - 8.f * a.slice(fp1, sz) + 8.f * a.slice(fm1, sz) - a.slice(fm2, sz)) / b.slice(f0, sz).constant(12.f);
  }
}
} // namespace

template <int ND>
Grad<ND>::Grad(InDims const ish, std::vector<Index> const &d, int const o)
  : Parent("Grad", ish, AddBack(ish, (Index)d.size()))
  , dims_{d}
  , order_{o}
{
}

template <int ND> void Grad<ND>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  y.setZero();
  for (Index ii = 0; ii < (Index)dims_.size(); ii++) {
    switch (order_) {
    case 0: ForwardDiff<true>(x, y.template chip<ND>(ii), x.dimensions(), dims_[ii]); break;
    case 1: CentralDiff0<true>(x, y.template chip<ND>(ii), x.dimensions(), dims_[ii]); break;
    case 2: CentralDiff1<true>(x, y.template chip<ND>(ii), x.dimensions(), dims_[ii]); break;
    }
  }
  this->finishForward(y, time, false);
}

template <int ND> void Grad<ND>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.setZero();
  for (Index ii = 0; ii < (Index)dims_.size(); ii++) {
    switch (order_) {
    case 0: ForwardDiff<false>(y.template chip<ND>(ii), x, x.dimensions(), dims_[ii]); break;
    case 1: CentralDiff0<false>(y.template chip<ND>(ii), x, x.dimensions(), dims_[ii]); break;
    case 2: CentralDiff1<false>(y.template chip<ND>(ii), x, x.dimensions(), dims_[ii]); break;
    }
    this->finishAdjoint(x, time, false);
  }
}

template struct Grad<5>;

template <int ND>
GradVec<ND>::GradVec(InDims const ishape, std::vector<Index> const &dims, int const o)
  : Parent("GradVec", ishape, AddBack(FirstN<ND - 1>(ishape), (Index)((dims.size() * (dims.size() + 1)) / 2)))
  , dims_{dims}
  , order_{o}
{
  if (dims.size() != ishape[ND - 1]) {
    throw(Log::Failure("gradv", "Symmetrized gradient only, dims were {} and {}", dims.size(), ishape[ND - 1]));
  }
}

template <int ND> void GradVec<ND>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  auto const sz = FirstN<ND - 1>(x.dimensions());
  y.setZero();
  /*
   * Grad applied to a vector produces a tensor. Here it is flattened back into a vector
   */
  Index yind = dims_.size();
  for (Index ii = 0; ii < dims_.size(); ii++) {
    switch (order_) {
    case 0: ForwardDiff<false>(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(ii), sz, dims_[ii]); break;
    case 1: CentralDiff0<false>(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(ii), sz, dims_[ii]); break;
    case 2: CentralDiff1<false>(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(ii), sz, dims_[ii]); break;
    }
    for (Index ij = ii + 1; ij < dims_.size(); ij++) {
      switch (order_) {
      case 0:
        ForwardDiff<false>(x.template chip<ND - 1>(ij), y.template chip<ND - 1>(yind), sz, dims_[ii]);
        ForwardDiff<false>(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(yind), sz, dims_[ij]);
        break;
      case 1:
        CentralDiff0<false>(x.template chip<ND - 1>(ij), y.template chip<ND - 1>(yind), sz, dims_[ii]);
        CentralDiff0<false>(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(yind), sz, dims_[ij]);
        break;
      case 2:
        CentralDiff1<false>(x.template chip<ND - 1>(ij), y.template chip<ND - 1>(yind), sz, dims_[ii]);
        CentralDiff1<false>(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(yind), sz, dims_[ij]);
        break;
      }
      y.template chip<ND - 1>(yind) /= y.template chip<ND - 1>(yind).constant(2.f);
      yind++;
    }
  }
  this->finishForward(y, time, false);
}

template <int ND> void GradVec<ND>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  auto const sz = FirstN<ND - 1>(x.dimensions());
  x.setZero();
  /*
   *  This is the tensor form of Div (see wikipedia page) but with the tensor flattened into a vector
   */
  Index yind = dims_.size();
  for (Index ii = 0; ii < dims_.size(); ii++) {
    switch (order_) {

    case 0: ForwardDiff<true>(y.template chip<ND - 1>(ii), x.template chip<ND - 1>(ii), sz, dims_[ii]); break;
    case 1: CentralDiff0<true>(y.template chip<ND - 1>(ii), x.template chip<ND - 1>(ii), sz, dims_[ii]); break;
    case 2: CentralDiff1<true>(y.template chip<ND - 1>(ii), x.template chip<ND - 1>(ii), sz, dims_[ii]); break;
    }
    for (Index ij = ii + 1; ij < dims_.size(); ij++) {
      switch (order_) {
      case 0:
        ForwardDiff<true>(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ii), sz, dims_[ij]);
        ForwardDiff<true>(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ij), sz, dims_[ii]);
        break;
      case 1:
        CentralDiff0<true>(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ii), sz, dims_[ij]);
        CentralDiff0<true>(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ij), sz, dims_[ii]);
        break;
      case 2:
        CentralDiff1<true>(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ii), sz, dims_[ij]);
        CentralDiff1<true>(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ij), sz, dims_[ii]);
        break;
      }
      yind++;
    }
  }
  this->finishAdjoint(x, time, false);
}

template struct GradVec<6>;

} // namespace rl::TOps
