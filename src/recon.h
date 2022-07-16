#pragma once

#include "op/recon.hpp"
#include "parse_args.h"
#include "sdc.h"
#include "sense.h"
#include "trajectory.h"

namespace rl {

auto make_sense(
  Trajectory const &traj,
  CoreOpts &core,
  ExtraOpts &extra,
  SENSE::Opts &sense,
  HD5::RieslingReader &reader,
  SDC::Opts &sdc,
  bool const toeplitz) -> std::unique_ptr<ReconOp>;

}
