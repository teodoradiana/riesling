#include "types.h"

#include "cropper.h"
#include "fft_plan.h"
#include "filter.h"
#include "io.h"
#include "log.h"
#include "op/grid.h"
#include "op/recon-rss.hpp"
#include "op/recon.hpp"
#include "parse_args.h"
#include "sdc.h"
#include "sense.h"
#include "tensorOps.h"

#include <variant>

int main_recon(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  COMMON_SENSE_ARGS;
  args::Flag rss(parser, "RSS", "Use Root-Sum-Squares channel combination", {"rss", 'r'});
  args::ValueFlag<float> sense_fov(parser, "F", "SENSE FoV (default 256mm)", {"sense_fov"}, 256);
  args::ValueFlag<std::string> basisFile(
    parser, "BASIS", "Read subspace basis from .h5 file", {"basis", 'b'});

  ParseCommand(parser, iname);
  FFT::Start();
  HD5::RieslingReader reader(iname.Get());
  Trajectory const traj = reader.trajectory();
  Info const &info = traj.info();

  auto const kernel = make_kernel(ktype.Get(), info.type, osamp.Get());
  auto const mapping = traj.mapping(kernel->inPlane(), osamp.Get());
  auto gridder = make_grid(kernel.get(), mapping, fastgrid);
  R2 const w = SDC::Choose(sdc.Get(), traj, osamp.Get());
  gridder->setSDC(w);
  gridder->setSDCPower(sdcPow.Get());

  std::unique_ptr<GridBase> bgridder = nullptr;
  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get());
    R2 const basis = basisReader.readTensor<R2>(HD5::Keys::Basis);
    bgridder = make_grid_basis(kernel.get(), mapping, basis, fastgrid);
    bgridder->setSDC(w);
  }

  std::variant<nullptr_t, ReconOp, ReconRSSOp> recon = nullptr;

  Sz4 sz;
  if (rss) {
    Cropper crop(info, gridder->mapping().cartDims, sense_fov); // To get correct dims
    recon.emplace<ReconRSSOp>(basisFile ? bgridder.get() : gridder.get(), crop.size());
    sz = std::get<ReconRSSOp>(recon).inputDimensions();
  } else {
    Cx4 senseMaps = sFile ? LoadSENSE(sFile.Get())
                          : SelfCalibration(
                              info,
                              gridder.get(),
                              sense_fov.Get(),
                              sRes.Get(),
                              sReg.Get(),
                              reader.noncartesian(ValOrLast(sVol.Get(), info.volumes)));
    recon.emplace<ReconOp>(basisFile ? bgridder.get() : gridder.get(), senseMaps);
    sz = std::get<ReconOp>(recon).inputDimensions();
  }

  Cropper out_cropper(info, LastN<3>(sz), out_fov.Get());
  Cx4 vol(sz);
  Sz3 outSz = out_cropper.size();
  Cx4 cropped(sz[0], outSz[0], outSz[1], outSz[2]);
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], info.volumes);
  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < info.volumes; iv++) {
    auto const &vol_start = Log::Now();
    if (rss) {
      vol = std::get<ReconRSSOp>(recon).Adj(reader.noncartesian(iv)); // Initialize
    } else {
      vol = std::get<ReconOp>(recon).Adj(reader.noncartesian(iv));
    }
    cropped = out_cropper.crop4(vol);
    out.chip<4>(iv) = cropped;
    Log::Print(FMT_STRING("Volume {}: {}"), iv, Log::ToNow(vol_start));
  }
  Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
  auto const fname = OutName(iname.Get(), oname.Get(), "recon", "h5");
  HD5::Writer writer(fname);
  writer.writeTrajectory(traj);
  writer.writeTensor(out, "image");
  FFT::End();
  return EXIT_SUCCESS;
}
