#include "types.h"

#include "apodizer.h"
#include "cropper.h"
#include "espirit.h"
#include "fft_plan.h"
#include "filter.h"
#include "gridder.h"
#include "io_hd5.h"
#include "io_nifti.h"
#include "log.h"
#include "parse_args.h"
#include "sense.h"

int main_espirit(args::Subparser &parser)
{
  CORE_RECON_ARGS;

  args::ValueFlag<long> volume(
      parser, "SENSE VOLUME", "Take SENSE maps from this volume (default last)", {"volume"}, -1);
  args::ValueFlag<float> fov(parser, "FOV", "FoV in mm (default header value)", {"fov"}, -1);
  args::ValueFlag<long> kRad(
      parser, "KERNEL RADIUS", "Kernel radius (default 6)", {"kernel", 'k'}, 4);
  args::ValueFlag<long> calRad(
      parser, "CAL RADIUS", "Additional calibration radius (default 1)", {"cal", 'c'}, 1);
  args::ValueFlag<float> retain(
      parser,
      "RETAIN",
      "ESPIRIT Fraction of singular vectors to retain (default 0.25)",
      {"retain"},
      0.25);
  args::ValueFlag<float> res(
      parser, "RESOLUTION", "Resolution for initial gridding (default 8 mm)", {"res", 'r'}, 8.f);
  args::Flag nifti(parser, "NIFTI", "Write output to nifti instead of .h5", {"nii"});

  Log log = ParseCommand(parser, fname);
  FFT::Start(log);

  HD5::Reader reader(fname.Get(), log);
  auto const traj = reader.readTrajectory();
  auto const &info = traj.info();
  Kernel *kernel =
      kb ? (Kernel *)new KaiserBessel(3, osamp.Get(), (info.type == Info::Type::ThreeD))
         : (Kernel *)new NearestNeighbour();

  Cx3 rad_ks = info.noncartesianVolume();
  reader.readNoncartesian(LastOrVal(volume, info.volumes), rad_ks);

  log.info(FMT_STRING("Cropping data to {} mm effective resolution"), res.Get());
  Cx3 lo_ks = rad_ks;
  auto const lo_traj = traj.trim(res.Get(), lo_ks);
  Gridder gridder(lo_traj, osamp.Get(), kernel, false, log);
  SDC::Load("pipe", lo_traj, gridder, log);
  long const totalCalRad =
      kRad.Get() + calRad.Get() + (gridder.info().spokes_lo ? 0 : gridder.info().read_gap);
  Cropper cropper(info, gridder.gridDims(), fov.Get(), log);
  Cx4 sense = cropper.crop4(ESPIRIT(gridder, lo_ks, kRad.Get(), totalCalRad, log));
  if (nifti) {
    WriteNifti(
        info,
        Cx4(sense.shuffle(Sz4{1, 2, 3, 0})),
        OutName(fname.Get(), oname.Get(), "espirit", "nii"),
        log);
  } else {
    HD5::Writer writer(OutName(fname.Get(), oname.Get(), "espirit", "h5"), log);
    writer.writeSENSE(sense);
  }

  FFT::End(log);
  return EXIT_SUCCESS;
}
