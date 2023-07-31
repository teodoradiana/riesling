#include "ndft.hpp"

#include "op/loop.hpp"
#include "op/rank.hpp"

using namespace std::complex_literals;

namespace rl {

template <size_t NDim>
NDFTOp<NDim>::NDFTOp(
  Re3 const &tr, Index const nC, Sz<NDim> const shape, Re2 const &b, std::shared_ptr<TensorOperator<Cx, 3>> s)
  : Parent("NDFTOp", AddFront(shape, nC, b.dimension(0)), AddFront(LastN<2>(tr.dimensions()), nC))
  , traj{tr}
  , basis{b}
  , sdc{s ? s : std::make_shared<TensorIdentity<Cx, 3>>(oshape)}
{
  static_assert(NDim < 4);
  if (traj.dimension(0) != NDim) { Log::Fail("Requested {}D NDFT but trajectory is {}D", NDim, traj.dimension(0)); }
  Log::Print<Log::Level::High>("NDFT Input Dims {} Output Dims {}", ishape, oshape);
  Log::Print<Log::Level::High>("Calculating cartesian co-ords");
  N = Product(shape);
  scale = 1.f / std::sqrt(N);
  Re1 trScale(NDim);
  for (Index ii = 0; ii < NDim; ii++) {
    trScale(ii) = shape[ii];
  }
  traj = traj * trScale.reshape(Sz3{NDim, 1, 1}).broadcast(Sz3{1, traj.dimension(1), traj.dimension(2)});
  xc.resize(NDim, N);
  xind.resize(N);
  Index ind = 0;
  Index const si = shape[NDim - 1];
  for (int16_t ii = 0; ii < si; ii++) {
    float const fi = (ii - (si - 1.f) / 2.f) / si;
    if constexpr (NDim == 1) {
      xind[ind] = {ii};
      xc(ind) = 2.f * M_PI * fi;
      ind++;
    } else {
      Index const sj = shape[NDim - 2];
      for (int16_t ij = 0; ij < sj; ij++) {
        float const fj = (ij - (sj - 1.f) / 2.f) / sj;
        if constexpr (NDim == 2) {
          xind[ind] = {ij, ii};
          xc.col(ind) = 2.f * M_PI * Eigen::Vector2f(fj, fi);
          ind++;
        } else {
          Index const sk = shape[NDim - 3];
          for (int16_t ik = 0; ik < sk; ik++) {
            float const fk = (ik - (sk - 1.f) / 2.f) / sk;
            xind[ind] = {ik, ij, ii};
            xc.col(ind) = 2.f * M_PI * Eigen::Vector3f(fk, fj, fi);
            ind++;
          }
        }
      }
    }
  }
}

template <size_t NDim>
void NDFTOp<NDim>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x);
  using RVec = typename Eigen::RowVector<float, NDim>;
  using FMap = typename Eigen::Matrix<float, NDim, -1>::ConstMapType;
  using CxMap = typename Eigen::Matrix<Cx, -1, 1>::MapType;
  using CxCMap = typename Eigen::Map<Eigen::Matrix<Cx, -1, -1> const, Eigen::Aligned32>;

  Index const nC = ishape[0];
  Index const nB = ishape[1];
  Index const nSamp = traj.dimension(1);
  Index const nTrace = traj.dimension(2);

  FMap   tm(traj.data(), NDim, nSamp * nTrace);
  CxCMap xm(x.data(), nC * nB, N);
  CxMap  ym(y.data(), nC * nB, nSamp * nTrace);

  auto task = [&](Index ii) {
    RVec const             f = -tm.col(ii).transpose();
    Eigen::VectorXf const  ph = f * xc;
    Eigen::VectorXcf const eph = ph.unaryExpr([](float const p) { return std::polar(1.f, p); });
    Eigen::VectorXcf const vox = xm * eph;
    ym.col(ii) = vox * scale;
  };

  Threads::For(task, nSamp * nTrace, "NDFT Forward");
  this->finishForward(y, time);
}

template <size_t NDim>
void NDFTOp<NDim>::adjoint(OutCMap const &yy, InMap &x) const
{
  auto const time = this->startAdjoint(yy);
  OutTensor  sy;
  OutCMap    y(yy);
  if (sdc) {
    sy.resize(yy.dimensions());
    sy = sdc->adjoint(yy);
    new (&y) OutCMap(sy.data(), sy.dimensions());
  }

  using RVec = typename Eigen::RowVector<float, NDim>;
  using FMap = typename Eigen::Matrix<float, NDim, -1>::ConstMapType;
  using CxMap = typename Eigen::Matrix<Cx, -1, 1>::MapType;
  using CxCMap = typename Eigen::Map<Eigen::Matrix<Cx, -1, -1> const, Eigen::Aligned32>;

  Index const nC = ishape[0];
  Index const nB = ishape[1];
  Index const nSamp = traj.dimension(1);
  Index const nTrace = traj.dimension(2);

  FMap   tm(traj.data(), NDim, nSamp * nTrace);
  CxCMap ym(y.data(), nC * nB, nSamp * nTrace);
  CxMap  xm(x.data(), nC * nB, N);

  auto task = [&](Index ii) {
    RVec const             f = xc.col(ii).transpose();
    Eigen::VectorXf const  ph = f * tm;
    Eigen::VectorXcf const eph = ph.unaryExpr([](float const p) { return std::polar(1.f, p); });

    Eigen::VectorXcf const vox = ym * eph;
    xm.col(ii) = vox * scale;
  };

  Threads::For(task, xind.size(), "NDFT Adjoint");
  this->finishAdjoint(x, time);
}

template struct NDFTOp<1>;
template struct NDFTOp<2>;
template struct NDFTOp<3>;

std::shared_ptr<TensorOperator<Cx, 5, 4>>
make_ndft(Re3 const &traj, Index const nC, Sz3 const matrix, Re2 const &basis, std::shared_ptr<TensorOperator<Cx, 3>> sdc)
{

  std::shared_ptr<TensorOperator<Cx, 5, 4>> ndft;
  if (traj.dimension(0) == 2) {
    Log::Print<Log::Level::Debug>("Creating 2D Multi-slice NDFT");
    auto ndft2 = std::make_shared<NDFTOp<2>>(traj, nC, FirstN<2>(matrix), basis, sdc);
    ndft = std::make_shared<LoopOp<NDFTOp<2>>>(ndft2, matrix[2]);
  } else {
    Log::Print<Log::Level::Debug>("Creating full 3D NDFT");
    ndft = std::make_shared<IncreaseOutputRank<NDFTOp<3>>>(std::make_shared<NDFTOp<3>>(traj, nC, matrix, basis, sdc));
  }
  return ndft;
}

} // namespace rl
