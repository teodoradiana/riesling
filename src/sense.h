#pragma once

#include "cropper.h"
#include "info.h"
#include "io.h"
#include "log.h"
#include "op/grid.h"
#include "parse_args.h"

#define COMMON_SENSE_ARGS                                                                          \
  args::ValueFlag<std::string> senseFile(                                                          \
    parser, "SENSE", "Read SENSE maps from specified .h5 file", {"sense", 's'});                   \
  args::ValueFlag<Index> senseVol(                                                                 \
    parser, "SENSE VOLUME", "Take SENSE maps from this volume", {"senseVolume"}, -1);              \
  args::ValueFlag<float> senseLambda(                                                              \
    parser, "LAMBDA", "SENSE regularization", {"lambda", 'l'}, 0.f);

// Helper function for getting a good volume to take SENSE maps from
Index ValOrLast(Index const val, Index const last);

/*!
 * Calculates a set of SENSE maps from non-cartesian data, assuming an oversampled central region
 */
Cx4 DirectSENSE(
  Info const &info,
  GridBase const *g,
  float const fov,
  float const lambda,
  Cx3 const &data,
  Log &log);

/*!
 * Loads a set of SENSE maps from a file
 */
Cx4 LoadSENSE(std::string const &calFile, Log &log);

/*!
 * Loads a set of SENSE maps from a file and interpolate them to correct dims
 */
Cx4 InterpSENSE(std::string const &file, Eigen::Array3l const dims, Log &log);
