#include "types.h"

#include "admm.hpp"
#include "cropper.h"
#include "fft_plan.h"
#include "filter.h"
#include "io.h"
#include "llr.h"
#include "log.h"
#include "op/grid-basis.h"
#include "op/grid.h"
#include "op/recon.h"
#include "op/sense.h"
#include "parse_args.h"
#include "sense.h"

int main_admm(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  COMMON_SENSE_ARGS;
  args::Flag toeplitz(parser, "T", "Use Töplitz embedding", {"toe", 't'});
  args::ValueFlag<float> iter_fov(parser, "F", "Iterations FoV (default 256mm)", {"iter_fov"}, 256);
  args::ValueFlag<std::string> basisFile(parser, "BASIS", "Read basis from file", {"basis", 'b'});
  args::ValueFlag<float> thr(parser, "T", "Threshold for termination (1e-10)", {"thresh"}, 1.e-10);
  args::ValueFlag<Index> lsq_its(parser, "ITS", "Inner iterations (8)", {"lsq_its"}, 8);
  args::ValueFlag<Index> admm_its(parser, "ITS", "Outer iterations (8)", {"admm_its"}, 8);
  args::ValueFlag<float> reg_lambda(parser, "L", "ADMM lambda (default 0.1)", {"reg"}, 0.1f);
  args::ValueFlag<float> reg_rho(parser, "R", "ADMM rho (default 0.1)", {"rho"}, 0.1f);
  args::ValueFlag<Index> patch(parser, "P", "Patch size for LLR (default 8)", {"patch"}, 8);

  Log log = ParseCommand(parser, iname);
  FFT::Start(log);

  HD5::Reader reader(iname.Get(), log);
  Trajectory const traj = reader.readTrajectory();
  Info const &info = traj.info();

  auto gridder = make_grid(traj, osamp.Get(), kernel.Get(), fastgrid, log);
  R2 const w = SDC::Choose(sdc.Get(), traj, osamp.Get(), log);
  gridder->setSDC(w);
  Cx4 senseMaps = senseFile ? LoadSENSE(senseFile.Get(), log)
                            : DirectSENSE(
                                info,
                                gridder.get(),
                                iter_fov.Get(),
                                senseLambda.Get(),
                                reader.noncartesian(ValOrLast(senseVol.Get(), info.volumes)),
                                log);

  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get(), log);
    R2 const basis = basisReader.readBasis();
    gridder = make_grid_basis(gridder->mapping(), kernel.Get(), fastgrid, basis, log);
    gridder->setSDC(w);
  }
  gridder->setSDCPower(sdcPow.Get());
  ReconOp recon(gridder.get(), senseMaps, log);
  if (!basisFile) {
    recon.calcToeplitz(traj.info());
  }

  auto reg = [&](Cx4 const &x) -> Cx4 { return llr(x, reg_lambda.Get(), patch.Get(), log); };

  auto sz = recon.inputDimensions();
  Cropper out_cropper(info, Last3(sz), out_fov.Get(), log);
  Cx4 vol(sz);
  Sz3 outSz = out_cropper.size();
  Cx4 cropped(sz[0], outSz[0], outSz[1], outSz[2]);
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], info.volumes);
  auto const &all_start = log.now();
  for (Index iv = 0; iv < info.volumes; iv++) {
    auto const &vol_start = log.now();
    recon.Adj(reader.noncartesian(iv), vol); // Initialize
    admm(admm_its.Get(), lsq_its.Get(), thr.Get(), recon, reg_rho.Get(), reg, vol, log);
    cropped = out_cropper.crop4(vol);
    out.chip<4>(iv) = cropped;
    log.info("Volume {}: {}", iv, log.toNow(vol_start));
  }
  log.info("All Volumes: {}", log.toNow(all_start));
  auto const fname = OutName(iname.Get(), oname.Get(), "admm", "h5");
  HD5::Writer writer(fname, log);
  writer.writeInfo(info);
  writer.writeTensor(out, "image");
  FFT::End(log);
  return EXIT_SUCCESS;
}
