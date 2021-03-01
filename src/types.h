#pragma once

#ifdef DEBUG
#define EIGEN_INITIALIZE_MATRICES_BY_NAN
#endif
// Need to define EIGEN_USE_THREADS before including these. This is done in CMakeLists.txt
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <complex>

using Array3l = Eigen::Array<long, 3, 1>;
using ArrayXl = Eigen::Array<long, -1, 1>;

using B0 = Eigen::TensorFixedSize<bool, Eigen::Sizes<>>;
using B3 = Eigen::Tensor<bool, 3>;

using L0 = Eigen::TensorFixedSize<long, Eigen::Sizes<>>;
using L1 = Eigen::Tensor<long, 1>;
using L2 = Eigen::Tensor<long, 2>;

using R0 = Eigen::TensorFixedSize<float, Eigen::Sizes<>>; // Annoying return type for reductions
using R1 = Eigen::Tensor<float, 1>;                       // 1D Real data
using R2 = Eigen::Tensor<float, 2>;                       // 2D Real data
using R3 = Eigen::Tensor<float, 3>;                       // 3D Real data
using R4 = Eigen::Tensor<float, 4>;                       // 4D Real data

using Rd1 = Eigen::Tensor<double, 1>;

using Cx = std::complex<float>;
using Cxd = std::complex<double>;

using Cx0 = Eigen::TensorFixedSize<Cx, Eigen::Sizes<>>;
using Cx1 = Eigen::Tensor<Cx, 1>; // 1D Complex data
using Cx2 = Eigen::Tensor<Cx, 2>; // 2D Complex data
using Cx3 = Eigen::Tensor<Cx, 3>; // 3D Complex data
using Cx4 = Eigen::Tensor<Cx, 4>; // 4D Complex data...spotted a pattern yet?
using Cx5 = Eigen::Tensor<Cx, 5>;
using Cx7 = Eigen::Tensor<Cx, 7>;

using Cxd1 = Eigen::Tensor<std::complex<double>, 1>; // 1D double precision complex data

// Useful shorthands
using Sz1 = Eigen::array<long, 1>;
using Sz2 = Eigen::array<long, 2>;
using Sz3 = Eigen::array<long, 3>;
using Sz4 = Eigen::array<long, 4>;
using Dims2 = Cx2::Dimensions;
using Dims3 = Cx3::Dimensions;
using Dims4 = Cx4::Dimensions;
using Size2 = Eigen::Array<int16_t, 2, 1>;
using Size3 = Eigen::Array<int16_t, 3, 1>;
using Size4 = Eigen::Array<int16_t, 4, 1>;
using Point2 = Eigen::Matrix<float, 2, 1>;
using Point3 = Eigen::Matrix<float, 3, 1>;
using Point4 = Eigen::Matrix<float, 4, 1>;
using Points3 = Eigen::Matrix<float, 3, -1>;
using Points4 = Eigen::Matrix<float, 4, -1>;
using Pads3 = Eigen::array<std::pair<long, long>, 3>;

// This is the type of the lambda functions to represent the encode/decode operators
using EncodeFunction = std::function<void(Cx3 &x, Cx3 &y)>;
using DecodeFunction = std::function<void(Cx3 const &x, Cx3 &y)>;
using SystemFunction = std::function<void(Cx3 const &x, Cx3 &y)>;

inline Cx4 SwapToChannelLast(Cx4 const &x)
{
  return Cx4(x.shuffle(Sz4{1, 2, 3, 0}));
}
