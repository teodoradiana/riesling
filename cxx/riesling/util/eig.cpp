#include "types.hpp"

#include "algo/eig.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "precon.hpp"
#include "sense/sense.hpp"
#include "tensors.hpp"
#include "threads.hpp"

using namespace rl;

void main_eig(args::Subparser &parser)
{
  CoreOpts               coreOpts(parser);
  GridOpts               gridOpts(parser);
  PreconOpts             preOpts(parser);
  SENSE::Opts            senseOpts(parser);
  args::Flag             adj(parser, "ADJ", "Use adjoint system AA'", {"adj"});
  args::ValueFlag<Index> its(parser, "N", "Max iterations (32)", {'i', "max-its"}, 40);
  args::Flag             recip(parser, "R", "Output reciprocal of eigenvalue", {"recip"});
  args::Flag             savevec(parser, "S", "Output the corresponding eigenvector", {"savevec"});
  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader, reader.readInfo().voxel_size);
  auto        noncart = reader.readTensor<Cx5>();
  auto const  basis = ReadBasis(coreOpts.basisFile.Get());
  auto const A = Recon::SENSE(coreOpts, gridOpts, senseOpts, traj, reader.dimensions()[3], basis, noncart);
  auto const P = make_kspace_pre(traj, A->oshape[0], ReadBasis(coreOpts.basisFile.Get()), gridOpts.vcc, preOpts.type.Get(),
                                 preOpts.bias.Get());

  if (adj) {
    auto const [val, vec] = PowerMethodAdjoint(A, P, its.Get());
    if (savevec) {
      HD5::Writer writer(coreOpts.oname.Get());
      writer.writeTensor("evec", A->ishape, vec.data());
    }
    fmt::print("{}\n", recip ? (1.f / val) : val);
  } else {
    auto const [val, vec] = PowerMethodForward(A, P, its.Get());
    if (savevec) {
      HD5::Writer writer(coreOpts.oname.Get());
      writer.writeTensor("evec", A->ishape, vec.data());
    }
    fmt::print("{}\n", recip ? (1.f / val) : val);
  }
}
