#pragma once

#include "io/hd5.hpp"
#include "log.hpp"
#include "op/gridBase.hpp"
#include "op/sdc.hpp"
#include "parse_args.hpp"
#include "trajectory.hpp"

namespace rl {
namespace SENSE {

struct Opts
{
  Opts(args::Subparser &parser);
  args::ValueFlag<std::string> file;
  args::ValueFlag<Index> volume, frame;
  args::ValueFlag<float> res, λ;
};

//! Calculates a set of SENSE maps from non-cartesian data, assuming an oversampled central region
Cx4 SelfCalibration(
  Trajectory const &traj, GridBase<Cx, 3> *g, float const fov, float const res, float const λ, Index const frame, Cx3 const &data);
Cx4 Interp(std::string const &calFile, Sz3 const dims); //! Interpolate with FFT

//! Convenience function called from recon commands to get SENSE maps
Cx4 Choose(Opts &opts, Trajectory const &t, GridBase<Cx, 3> *gridder, float const fov, SDCOp *sdc, HD5::RieslingReader &reader);

} // namespace SENSE
} // namespace rl
