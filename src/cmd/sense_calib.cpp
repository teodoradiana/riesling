#include "types.h"

#include "io/hd5.hpp"
#include "log.h"
#include "op/gridBase.hpp"
#include "parse_args.h"
#include "sdc.h"
#include "sense.h"
#include "fft/fft.hpp"
#include "cropper.h"

using namespace rl;

int main_sense_calib(args::Subparser &parser)
{
  CoreOpts core(parser);
  SDC::Opts sdcOpts(parser);
  args::ValueFlag<Index> volume(parser, "V", "SENSE calibration volume", {"sense-vol"}, -1);
  args::ValueFlag<Index> frame(parser, "F", "SENSE calibration frame", {"sense-frame"}, 0);
  args::ValueFlag<float> res(parser, "R", "SENSE calibration res (12 mm)", {"sense-res"}, 12.f);
  args::ValueFlag<float> λ(parser, "L", "SENSE regularization", {"sense-lambda"}, 0.f);
  args::ValueFlag<float> fov(parser, "FOV", "FoV in mm (default 256 mm)", {"fov"}, 256.f);
  args::ValueFlag<Index> kernels(parser, "K", "Save kernels with size K instead of maps", {"kernels"});
  ParseCommand(parser, core.iname);

  HD5::RieslingReader reader(core.iname.Get());
  auto const traj = reader.trajectory();
  auto const &info = traj.info();
  auto const kernel = rl::make_kernel(core.ktype.Get(), info.type, core.osamp.Get());
  Mapping const mapping(reader.trajectory(), kernel.get(), core.osamp.Get(), core.bucketSize.Get());
  auto gridder = make_grid<Cx>(kernel.get(), mapping, info.channels, core.basisFile.Get());
  auto const sdc = SDC::Choose(sdcOpts, traj, core.osamp.Get());
  Cx3 const data = sdc->Adj(reader.noncartesian(ValOrLast(volume.Get(), info.volumes)));
  
  auto const fname = OutName(core.iname.Get(), core.oname.Get(), "sense", "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(info);
  if (kernels) {
    Cx4 const ksense = SENSE::SelfCalibrationKernels(info, gridder.get(), res.Get(), λ.Get(), frame.Get(), kernels.Get(), data);
    writer.writeTensor(ksense, HD5::Keys::Kernels);
  } else {
    Cx4 sense = SENSE::SelfCalibration(info, gridder.get(), fov.Get(), res.Get(), λ.Get(), frame.Get(), data);
    writer.writeTensor(sense, HD5::Keys::SENSE);
  }

  return EXIT_SUCCESS;
}
