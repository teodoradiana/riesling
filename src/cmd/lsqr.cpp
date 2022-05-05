#include "types.h"

#include "algo/lsqr.hpp"
#include "log.h"
#include "op/recon.hpp"
#include "parse_args.h"
#include "precond/single.hpp"
#include "sdc.h"
#include "sense.h"
#include "tensorOps.h"
#include "threads.h"

int main_lsqr(args::Subparser &parser)
{
  CoreOpts core(parser);
  ExtraOpts extra(parser);
  SDC::Opts sdcOpts(parser);
  SENSE::Opts senseOpts(parser);
  args::ValueFlag<std::string> basisFile(parser, "BASIS", "Read basis from file", {"basis", 'b'});
  args::ValueFlag<Index> its(parser, "N", "Max iterations (8)", {'i', "max-its"}, 8);
  args::ValueFlag<Index> readStart(parser, "N", "Read start", {"rs"}, 0);
  args::Flag precond(parser, "P", "Apply Ong's single-channel pre-conditioner", {"pre"});
  args::ValueFlag<float> atol(parser, "A", "Tolerance on A (1e-6)", {"atol"}, 1.e-6f);
  args::ValueFlag<float> btol(parser, "B", "Tolerance on b (1e-6)", {"btol"}, 1.e-6f);
  args::ValueFlag<float> ctol(parser, "C", "Tolerance on cond(A) (1e-6)", {"ctol"}, 1.e-6f);
  args::ValueFlag<float> damp(parser, "λ", "Tikhonov parameter (default 0)", {"lambda"}, 0.f);

  ParseCommand(parser, core.iname);

  HD5::RieslingReader reader(core.iname.Get());
  Trajectory const traj = reader.trajectory();
  Info const &info = traj.info();

  auto const kernel = make_kernel(core.ktype.Get(), info.type, core.osamp.Get());
  auto const mapping = traj.mapping(kernel->inPlane(), core.osamp.Get(), 0, readStart.Get());
  auto gridder = make_grid(kernel.get(), mapping, core.fast);

  std::unique_ptr<Precond<Cx3>> pre = precond.Get() ? std::make_unique<SingleChannel>(traj, kernel.get()) : nullptr;
  auto const sdc = SDC::Choose(sdcOpts, traj, core.osamp.Get());
  Cx4 senseMaps = SENSE::Choose(senseOpts, info, gridder.get(), extra.iter_fov.Get(), sdc.get(), reader);

  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get());
    R2 const basis = basisReader.readTensor<R2>(HD5::Keys::Basis);
    gridder = make_grid_basis(kernel.get(), gridder->mapping(), basis, core.fast);
  }
  ReconOp recon(gridder.get(), senseMaps, nullptr);

  auto sz = recon.inputDimensions();
  Cropper out_cropper(info, LastN<3>(sz), extra.out_fov.Get());
  Cx4 vol(sz);
  Sz3 outSz = out_cropper.size();
  Cx4 cropped(sz[0], outSz[0], outSz[1], outSz[2]);
  Cx5 out(sz[0], outSz[0], outSz[1], outSz[2], info.volumes);

  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < info.volumes; iv++) {
    auto const &vol_start = Log::Now();
    vol =
      lsqr(its.Get(), recon, reader.noncartesian(iv), atol.Get(), btol.Get(), ctol.Get(), damp.Get(), pre.get(), true);
    cropped = out_cropper.crop4(vol);
    out.chip<4>(iv) = cropped;
    Log::Print(FMT_STRING("Volume {}: {}"), iv, Log::ToNow(vol_start));
  }
  Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
  auto const fname = OutName(core.iname.Get(), core.oname.Get(), "lsqr", "h5");
  HD5::Writer writer(fname);
  writer.writeTrajectory(traj);
  writer.writeTensor(out, "image");

  return EXIT_SUCCESS;
}
