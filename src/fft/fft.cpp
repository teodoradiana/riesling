#include "fft.hpp"

#include "cpu.hpp"
#ifdef RL_CUDA
#include "cuda.hpp"
#endif

#include "../log.h"
#include "../tensorOps.h"
#include "../threads.h"
#include "fftw3.h"
#include <filesystem>
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>

namespace FFT {

enum struct Type
{
  FFTW = 0
#ifdef RL_CUDA
  ,
  CUDA
#endif
};

namespace {
Type type = Type::FFTW;
}

std::filesystem::path WisdomPath()
{
  struct passwd *pw = getpwuid(getuid());
  const char *homedir = pw->pw_dir;
  return std::filesystem::path(homedir) / ".riesling-wisdom";
}

void Init(std::string const &choice)
{
  if (choice == "fftw") {
    type = Type::FFTW;
    fftwf_init_threads();
    fftwf_make_planner_thread_safe();
    fftwf_set_timelimit(60.0);
    auto const wp = WisdomPath();
    if (fftwf_import_wisdom_from_filename(WisdomPath().string().c_str())) {
      Log::Print(FMT_STRING("Read wisdom successfully from {}"), wp);
    } else {
      Log::Print(FMT_STRING("Could not read wisdom from {}, continuing"), wp);
    }
  }
#ifdef RL_CUDA
  else if (choice == "cuda") {
    type = Type::CUDA;
  }
#endif
  else {
    Log::Fail(FMT_STRING("Unknown FFT type: {}"), choice);
  }
}

void End()
{
  if (type == Type::FFTW) {
    auto const &wp = WisdomPath();
    if (fftwf_export_wisdom_to_filename(wp.string().c_str())) {
      Log::Print(FMT_STRING("Saved wisdom to {}"), wp);
    } else {
      Log::Print(FMT_STRING("Failed to save wisdom to {}"), wp);
    }
    // Causes use after free errors if this is called before fftw_plan_destroy in the
    // destructors. We don't stop and re-start FFTW threads so calling this is not essential
    fftwf_cleanup_threads();
  }
}

void SetTimelimit(double time)
{
  if (type == Type::FFTW) {
    fftwf_set_timelimit(time);
    Log::Debug(FMT_STRING("Set FFT planning timelimit to {} seconds"), time);
  }
}

/*
 * Phase factors for FFT shifting
 *
 * I am indebted to Martin Uecker for putting this code in BART
 */
Cx1 Phase(Index const sz)
{
  Index const c = sz / 2;
  double const shift = (double)c / sz;
  Rd1 ii(sz);
  std::iota(ii.data(), ii.data() + ii.size(), 0.);
  auto const s = ((ii - ii.constant(c / 2.)) * ii.constant(shift));
  Cxd1 const ph = ((s - s.floor()) * s.constant(2. * M_PI)).cast<Cxd>();
  Cx1 const factors = (ph * ph.constant(Cxd{0., 1.})).exp().cast<Cx>();
  Log::Debug(FMT_STRING("Calculated FFT Phase factors length {}"), sz);
  return factors;
}

template <int TRank, int FFTRank>
std::unique_ptr<FFT<TRank, FFTRank>>
Make(typename FFT<TRank, FFTRank>::TensorDims const &dims, Index const inThreads)
{
  if (inThreads > 0) {
    return std::make_unique<CPU<TRank, FFTRank>>(dims, inThreads);
  } else {
    if (type == Type::FFTW) {
      return std::make_unique<CPU<TRank, FFTRank>>(dims, Threads::GlobalThreadCount());
    }
#ifdef RL_CUDA
    else if (type == Type::CUDA) {
      return std::make_unique<CUDAFFT<TRank, FFTRank>>(dims);
    }
#endif
    __builtin_unreachable();
  }
}

template std::unique_ptr<FFT<3, 3>> Make(typename FFT<3, 3>::TensorDims const &, Index const);
template std::unique_ptr<FFT<4, 3>> Make(typename FFT<4, 3>::TensorDims const &, Index const);
template std::unique_ptr<FFT<4, 1>> Make(typename FFT<4, 3>::TensorDims const &, Index const);
template std::unique_ptr<FFT<5, 3>> Make(typename FFT<5, 3>::TensorDims const &, Index const);
template std::unique_ptr<FFT<3, 2>> Make(typename FFT<3, 2>::TensorDims const &, Index const);

} // namespace FFT