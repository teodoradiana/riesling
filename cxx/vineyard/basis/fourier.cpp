#include "fourier.hpp"

#include "../fft.hpp"
#include "io/writer.hpp"
#include "pad.hpp"

namespace rl {

FourierBasis::FourierBasis(Index const N, Index const samples, Index const traces, float const os)
{
  Cx3 eye(N, samples > 1 ? N : 1, traces > 1 ? N : 1);
  eye.setZero();
  float const energy = std::sqrt(samples * traces / (float)N);
  for (Index ii = 0; ii < N; ii++) {
    eye(ii, samples > 1 ? ii : 0, traces > 1 ? ii : 0) = Cx(energy);
  }
  Cx3 padded = Pad(eye, Sz3{N, (Index)(os * samples), (Index)(os * traces)});
  FFT::Adjoint(padded, Sz2{1, 2});
  basis = Crop(padded, Sz3{N, samples, traces});
}

} // namespace rl