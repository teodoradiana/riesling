#pragma once

#include "fft.hpp"

#include "../log.h"
#include "../tensorOps.h"

#include <cuda_runtime.h>
#include <cufftXt.h>

namespace FFT {

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                                         \
  {                                                                                                \
    auto status = static_cast<cudaError_t>(call);                                                  \
    if (status != cudaSuccess)                                                                     \
      Log::Fail(                                                                                   \
        FMT_STRING("CUDA RT failure {}, file {}, line {}: {} "),                                   \
        status,                                                                                    \
        __FILE__,                                                                                  \
        __LINE__,                                                                                  \
        cudaGetErrorString(status));                                                               \
  }
#endif

#ifndef CUFFT_CALL
#define CUFFT_CALL(call)                                                                           \
  {                                                                                                \
    auto status = static_cast<cufftResult>(call);                                                  \
    if (status != CUFFT_SUCCESS)                                                                   \
      Log::Fail(FMT_STRING("CUFFT failure {}, file {}, line {}"), status, __FILE__, __LINE__);     \
  }
#endif // CUFFT_CALL

template <int TRank, int FRank>
struct CUDAFFT final : FFT<TRank, FRank>
{
  using Tensor = typename FFT<TRank, FRank>::Tensor;
  using TensorDims = typename Tensor::Dimensions;

  CUDAFFT(TensorDims const &dims)
  {
    Eigen::GpuDevice dev(&stream_);
    // CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    std::array<int, FRank> sizes;
    int N = 1;
    int Nvox = 1;
    // Process the two different kinds of dimensions - howmany / FFT
    {
      constexpr int FStart = TRank - FRank;
      int ii = 0;
      for (; ii < FStart; ii++) {
        N *= dims[ii];
      }
      for (; ii < TRank; ii++) {
        int const sz = dims[ii];
        Nvox *= sz;
        sizes[ii - FStart] = sz;
        Cx1 const ph = Phase(sz); // Prep FFT phase factors
        auto const nb = ph.size() * sizeof(cufftComplex);
        phase_[ii - FStart] = static_cast<cufftComplex *>(dev.allocate(nb));
        dev.memcpyHostToDevice(phase_[ii - FStart], ph.data(), nb);
      }
    }
    scale_ = 1. / sqrt(Nvox);

    // Allocate GPU memory
    std::size_t Nbytes = (N * Nvox) * sizeof(cufftComplex);
    ptr_ = static_cast<cufftComplex *>(dev.allocate(Nbytes));

    // FFTW is row-major. Reverse dims as per
    // http://www.fftw.org/fftw3_doc/Column_002dmajor-Format.html#Column_002dmajor-Format
    std::reverse(sizes.begin(), sizes.end());
    auto const start = Log::Now();
    CUFFT_CALL(cufftCreate(&plan_));
    CUFFT_CALL(cufftPlanMany(
      &plan_, FRank, sizes.data(), sizes.data(), N, 1, sizes.data(), N, 1, CUFFT_C2C, N));
    CUFFT_CALL(cufftSetStream(plan_, stream_.stream()));
    Log::Debug(FMT_STRING("CUDA FFT planning took {}"), Log::ToNow(start));
  }

  ~CUDAFFT()
  {
    CUFFT_CALL(cufftDestroy(plan_));
  }

  void forward(Tensor &x) const //!< Image space to k-space
  {
    for (Index ii = 0; ii < TRank; ii++) {
      assert(x.dimension(ii) == dims_[ii]);
    }
    Log::Debug(FMT_STRING("Forward FFT"));
    Eigen::GpuDevice dev(&stream_);
    auto const start = Log::Now();
    auto const nb = x.size() * sizeof(Cx);
    dev.memcpyHostToDevice(ptr_, x.data(), nb);
    applyPhase(1.f, true);
    CUFFT_CALL(cufftExecC2C(plan_, ptr_, ptr_, CUFFT_FORWARD));
    applyPhase(scale_, true);
    dev.memcpyDeviceToHost(x.data(), ptr_, nb);
    dev.synchronize();
    Log::Debug(FMT_STRING("Forward FFT: {}"), Log::ToNow(start));
  }

  void reverse(Tensor &x) const //!< K-space to image space
  {
    for (Index ii = 0; ii < TRank; ii++) {
      assert(x.dimension(ii) == dims_[ii]);
    }
    Log::Debug(FMT_STRING("Reverse FFT"));
    Eigen::GpuDevice dev(&stream_);
    auto start = Log::Now();
    auto const nb = x.size() * sizeof(Cx);
    dev.memcpyHostToDevice(ptr_, x.data(), nb);
    applyPhase(scale_, false);
    CUFFT_CALL(cufftExecC2C(plan_, ptr_, ptr_, CUFFT_INVERSE));
    applyPhase(1.f, false);
    dev.memcpyDeviceToHost(x.data(), ptr_, nb);
    dev.synchronize();
    Log::Debug(FMT_STRING("Reverse FFT: {}"), Log::ToNow(start));
  }

private:
  void applyPhase(float const scale, bool const forward) const
  {
    Eigen::GpuDevice dev(&stream_);
    auto x = Eigen::TensorMap<Eigen::Tensor<cufftComplex, TRank>>(ptr_, dims_);
    constexpr int FStart = TRank - FRank;
    for (Index ii = 0; ii < FRank; ii++) {
      auto ph = Eigen::TensorMap<Eigen::Tensor<cufftComplex, 1>>(phase_[ii], dims_[ii]);
      Eigen::array<Index, TRank> rsh, brd;
      for (Index in = 0; in < TRank; in++) {
        rsh[in] = 1;
        brd[in] = x.dimension(in);
      }
      rsh[FStart + ii] = dims_[ii];
      brd[FStart + ii] = 1;

      if (forward) {
        x.device(dev) = x * ph.reshape(rsh).broadcast(brd);
      } else {
        x.device(dev) = x / ph.reshape(rsh).broadcast(brd);
      }
    }
    if (scale != 1.f) {
      cuComplex cscale;
      cscale.x = scale;
      cscale.y = 0.f;
      x.device(dev) = x * x.constant(cscale);
    }
  }

  TensorDims dims_;
  Eigen::GpuStreamDevice stream_;
  cufftComplex *ptr_;

  cufftHandle plan_;
  std::array<cufftComplex *, 3> phase_;
  float scale_;
};

} // namespace FFT
