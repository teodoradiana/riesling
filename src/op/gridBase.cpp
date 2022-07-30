#include "gridBase.hpp"
#include "io/reader.hpp"

namespace rl {

// Forward Declare
template <int IP, int TP, typename Scalar>
std::unique_ptr<GridBase<Scalar>> make_grid_internal(Kernel const *k, Mapping const &m, Index const nC);
template <int IP, int TP, typename Scalar>
std::unique_ptr<GridBase<Scalar>>
make_grid_internal(Kernel const *k, Mapping const &m, Index const nC, R2 const &basis);

template <typename Scalar>
std::unique_ptr<GridBase<Scalar>>
make_grid(Kernel const *k, Mapping const &m, Index const nC, std::string const &basisFile)
{
  if (!basisFile.empty()) {
    HD5::Reader basisReader(basisFile);
    R2 const b = basisReader.readTensor<R2>(HD5::Keys::Basis);
    switch (k->inPlane()) {
    case 1:
      return make_grid_internal<1, 1, Scalar>(k, m, nC, b);
    case 3:
      if (k->throughPlane() == 1) {
        return make_grid_internal<3, 1, Scalar>(k, m, nC, b);
      } else {
        return make_grid_internal<3, 3, Scalar>(k, m, nC, b);
      }
      break;
    case 5:
      if (k->throughPlane() == 1) {
        return make_grid_internal<5, 1, Scalar>(k, m, nC, b);
      } else {
        return make_grid_internal<5, 5, Scalar>(k, m, nC, b);
      }
      break;
    case 7:
      if (k->throughPlane() == 1) {
        return make_grid_internal<7, 1, Scalar>(k, m, nC, b);
      } else {
        return make_grid_internal<7, 7, Scalar>(k, m, nC, b);
      }
      break;
    }
  } else {
    switch (k->inPlane()) {
    case 1:
      return make_grid_internal<1, 1, Scalar>(k, m, nC);
    case 3:
      if (k->throughPlane() == 1) {
        return make_grid_internal<3, 1, Scalar>(k, m, nC);
      } else {
        return make_grid_internal<3, 3, Scalar>(k, m, nC);
      }
      break;
    case 5:
      if (k->throughPlane() == 1) {
        return make_grid_internal<5, 1, Scalar>(k, m, nC);
      } else {
        return make_grid_internal<5, 5, Scalar>(k, m, nC);
      }
      break;
    case 7:
      if (k->throughPlane() == 1) {
        return make_grid_internal<7, 1, Scalar>(k, m, nC);
      } else {
        return make_grid_internal<7, 7, Scalar>(k, m, nC);
      }
      break;
    }
  }
  Log::Fail(FMT_STRING("No grids implemented for in-plane kernel width {}"), k->inPlane());
}

template std::unique_ptr<GridBase<float>>
make_grid<float>(Kernel const *k, Mapping const &m, Index const nC, std::string const &basisFile);
template std::unique_ptr<GridBase<Cx>>
make_grid<Cx>(Kernel const *k, Mapping const &m, Index const nC, std::string const &basisFile);

template <int IP, int TP>
auto make_decanter_internal(Kernel const *k, Mapping const &m, Cx4 const &kS) -> std::unique_ptr<GridBase<Cx>>;
template <int IP, int TP>
auto make_decanter_internal(Kernel const *k, Mapping const &m, Cx4 const &kS, R2 const &basis)
  -> std::unique_ptr<GridBase<Cx>>;

std::unique_ptr<GridBase<Cx>>
make_decanter(Kernel const *k, Mapping const &m, Cx4 const &kS, std::string const &basisFile)
{
  if (!basisFile.empty()) {
    HD5::Reader basisReader(basisFile);
    R2 const b = basisReader.readTensor<R2>(HD5::Keys::Basis);
    switch (k->inPlane()) {
    case 1:
      return make_decanter_internal<1, 1>(dynamic_cast<SizedKernel<1, 1> const *>(k), m, kS, b);
    case 3:
      if (k->throughPlane() == 1) {
        return make_decanter_internal<3, 1>(dynamic_cast<SizedKernel<3, 1> const *>(k), m, kS, b);
      } else if (k->throughPlane() == 3) {
        return make_decanter_internal<3, 3>(dynamic_cast<SizedKernel<3, 3> const *>(k), m, kS, b);
      }
      break;
    case 5:
      if (k->throughPlane() == 1) {
        return make_decanter_internal<5, 1>(dynamic_cast<SizedKernel<5, 1> const *>(k), m, kS, b);
      } else if (k->throughPlane() == 5) {
        return make_decanter_internal<5, 5>(dynamic_cast<SizedKernel<5, 5> const *>(k), m, kS, b);
      }
      break;
    }
  } else {
    switch (k->inPlane()) {
    case 1:
      return make_decanter_internal<1, 1>(dynamic_cast<SizedKernel<1, 1> const *>(k), m, kS);
    case 3:
      if (k->throughPlane() == 1) {
        return make_decanter_internal<3, 1>(dynamic_cast<SizedKernel<3, 1> const *>(k), m, kS);
      } else if (k->throughPlane() == 3) {
        return make_decanter_internal<3, 3>(dynamic_cast<SizedKernel<3, 3> const *>(k), m, kS);
      }
      break;
    case 5:
      if (k->throughPlane() == 1) {
        return make_decanter_internal<5, 1>(dynamic_cast<SizedKernel<5, 1> const *>(k), m, kS);
      } else if (k->throughPlane() == 5) {
        return make_decanter_internal<5, 5>(dynamic_cast<SizedKernel<5, 5> const *>(k), m, kS);
      }
      break;
    }
  }
  Log::Fail(FMT_STRING("No grids implemented for in-plane kernel width {}"), k->inPlane());
}

} // namespace rl
