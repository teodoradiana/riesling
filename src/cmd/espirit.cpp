#include "types.h"

#include "cropper.h"
#include "espirit.h"
#include "io/hd5.hpp"
#include "log.hpp"
#include "mapping.hpp"
#include "op/gridBase.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "sense.h"
#include "tensorOps.hpp"

using namespace rl;

int main_espirit(args::Subparser &parser)
{
  CoreOpts core(parser);
  SDC::Opts sdcOpts(parser);
  args::ValueFlag<Index> volume(parser, "VOL", "Take SENSE maps from this volume (default first)", {"sense-vol"}, 0);
  args::ValueFlag<float> res(parser, "R", "Resolution for initial gridding (default 12 mm)", {"sense-res", 'r'}, 12.f);
  args::ValueFlag<float> fov(parser, "FOV", "FoV in mm (default header value)", {"fov"}, -1);
  args::ValueFlag<Index> lores(parser, "L", "Lo-res traces", {"lores"}, 0);
  args::ValueFlag<Index> readStart(parser, "R", "Reference region start (0)", {"read-start"}, 0);
  args::ValueFlag<Index> kRad(parser, "RAD", "Kernel radius (default 4)", {"krad", 'k'}, 4);
  args::ValueFlag<Index> calRad(parser, "RAD", "Additional calibration radius (default 1)", {"calRad", 'c'}, 1);
  args::ValueFlag<float> thresh(parser, "T", "Variance threshold to retain kernels (0.015)", {"thresh"}, 0.015);

  ParseCommand(parser, core.iname);

  HD5::RieslingReader reader(core.iname.Get());
  auto const traj = reader.trajectory();
  auto const ks1 = reader.noncartesian(volume.Get());
  auto const &info = traj.info();
  Log::Print(FMT_STRING("Cropping data to {} mm effective resolution"), res.Get());
  auto const [dsTraj, ks] = traj.downsample(ks1, res.Get(), lores.Get(), false);
  auto const dsInfo = dsTraj.info();
  auto gridder = make_grid<Cx, 3>(dsTraj, core.ktype.Get(), core.osamp.Get(), ks.dimension(0));
  auto const sdc = SDC::Choose(sdcOpts, dsTraj, ks1.dimension(0), core.ktype.Get(), core.osamp.Get());
  Index const totalCalRad = kRad.Get() + calRad.Get() + readStart.Get();
  Cropper cropper(traj.info().matrix, LastN<3>(gridder->inputDimensions()), traj.info().voxel_size, fov.Get());
  Cx4 sense =
    cropper.crop4(ESPIRIT(gridder.get(), sdc->adjoint(ks), kRad.Get(), totalCalRad, readStart.Get(), thresh.Get()));

  auto const fname = OutName(core.iname.Get(), core.oname.Get(), "espirit", "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(info);
  writer.writeTensor(sense, HD5::Keys::SENSE);

  return EXIT_SUCCESS;
}
