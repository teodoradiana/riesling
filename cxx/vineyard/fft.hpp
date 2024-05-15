#pragma once

#include "types.hpp"

namespace rl {

namespace FFT {

template <int ND, int NFFT>
void Forward(CxN<ND> &data, Sz<NFFT> const fftDims, Index const threads = 0);
template <int ND>
void Forward(CxN<ND> &data, Index const threads = 0);

template <int ND, int NFFT>
void Forward(Eigen::TensorMap<CxN<ND>> &data, Sz<NFFT> const fftDims, Index const threads = 0);
template <int ND>
void Forward(Eigen::TensorMap<CxN<ND>> &data, Index const threads = 0);

template <int ND, int NFFT>
void Adjoint(Eigen::TensorMap<CxN<ND>> &data, Sz<NFFT> const fftDims, Index const threads = 0);
template <int ND>
void Adjoint(Eigen::TensorMap<CxN<ND>> &data, Index const threads = 0);

template <int ND, int NFFT>
void Adjoint(CxN<ND> &data, Sz<NFFT> const fftDims, Index const threads = 0);
template <int ND>
void Adjoint(CxN<ND> &data, Index const threads = 0);

/*
 * Phase ramps for FFT shifting
 */
template <int NFFT>
auto PhaseShift(Sz<NFFT> const shape) -> CxN<NFFT>;
} // namespace FFT

} // namespace rl
