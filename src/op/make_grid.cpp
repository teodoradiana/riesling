#include "make_grid.hpp"
#include "grid.hpp"
#include "io/reader.hpp"
#include "kernel/nn.hpp"
#include "loop.hpp"
#include "rank.hpp"

namespace rl {

auto IdBasis() -> Re2 {
  Re2 id(1, 1);
  id.setConstant(1.f);
  return id;
}

// Forward Declare
template <typename Scalar, size_t ND>
auto make_kb_radial(
  Trajectory const &traj, size_t const W, float const osamp, Index const nC, Re2 const &basis)
  -> std::shared_ptr<GridBase<Scalar, ND>>;
template <typename Scalar, size_t ND>
auto make_es_radial(
  Trajectory const &traj, size_t const W, float const osamp, Index const nC, Re2 const &basis)
  -> std::shared_ptr<GridBase<Scalar, ND>>;
template <typename Scalar, size_t ND>
auto make_kb_rect(
  Trajectory const &traj, size_t const W, float const osamp, Index const nC, Re2 const &basis)
  -> std::shared_ptr<GridBase<Scalar, ND>>;
template <typename Scalar, size_t ND>
auto make_es_rect(
  Trajectory const &traj, size_t const W, float const osamp, Index const nC, Re2 const &basis)
  -> std::shared_ptr<GridBase<Scalar, ND>>;

template <typename Scalar, size_t ND>
auto make_grid(
  Trajectory const &traj, std::string const kType, float const osamp, Index const nC, Re2 const &basis)
  -> std::shared_ptr<GridBase<Scalar, ND>>
{
  if (kType == "NN") {
    return std::make_shared<Grid<Scalar, NearestNeighbour<ND>>>(Mapping<ND>(traj, osamp, 1), nC, basis);
  } else if (kType.size() == 7 && kType.substr(0, 4) == "rect") {
    std::string const type = kType.substr(4, 2);
    size_t const W = std::stoi(kType.substr(6, 1));
    if (type == "ES") {
      return make_es_rect<Scalar, ND>(traj, W, osamp, nC, basis);
    } else if (type == "KB") {
      return make_kb_rect<Scalar, ND>(traj, W, osamp, nC, basis);
    }
  } else if (kType.size() == 3) {
    std::string const type = kType.substr(0, 2);
    size_t const W = std::stoi(kType.substr(2, 1));
    if (type == "ES") {
      return make_es_radial<Scalar, ND>(traj, W, osamp, nC, basis);
    } else if (type == "KB") {
      return make_kb_radial<Scalar, ND>(traj, W, osamp, nC, basis);
    }
  }
  Log::Fail("Invalid kernel type {}", kType);
}

template std::shared_ptr<GridBase<Cx, 2>>
make_grid<Cx, 2>(Trajectory const &, std::string const, float const, Index const, Re2 const &);
template std::shared_ptr<GridBase<float, 2>>
make_grid<float, 2>(Trajectory const &, std::string const, float const, Index const, Re2 const &);
template std::shared_ptr<GridBase<Cx, 3>>
make_grid<Cx, 3>(Trajectory const &, std::string const, float const, Index const, Re2 const &);
template std::shared_ptr<GridBase<float, 3>>
make_grid<float, 3>(Trajectory const &, std::string const, float const, Index const, Re2 const &);

std::shared_ptr<TensorOperator<Cx, 5, 4>> make_3d_grid(
  Trajectory const &traj,
  std::string const kType,
  float const os,
  Index const nC,
  Re2 const &basis)
{
  if (traj.nDims() == 2) {
    Log::Print<Log::Level::Debug>("Creating 2D Multi-slice Gridder");
    auto grid = make_grid<Cx, 2>(traj, kType, os, nC, basis);
    return std::make_shared<LoopOp<GridBase<Cx, 2>>>(grid, traj.info().matrix[2]);
  } else {
    Log::Print<Log::Level::Debug>("Creating 3D Gridder");
    return std::make_shared<IncreaseOutputRank<GridBase<Cx, 3>>>(make_grid<Cx, 3>(traj, kType, os, nC, basis));
  }
}

} // namespace rl
