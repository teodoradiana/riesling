#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "tensors.hpp"
#include "threads.hpp"

#include <scn/scan.h>

using namespace rl;

void main_maths(args::Subparser &parser)
{
  args::Positional<std::string> iname(parser, "FILE", "Input HD5 file");
  args::PositionalList<std::string> opsList(parser, "OPS", "Algebra operations");

  ParseCommand(parser, iname);
  HD5::Reader reader(iname.Get());

  auto ops = opsList.Get();

  if (ops.size() == 0) {
    Log::Fail("No operations specified");
  }

  auto const oname = ops.back();
  ops.pop_back();

  HD5::Writer writer(oname);
  writer.writeInfo(reader.readInfo());

  auto const order = reader.order();

  switch (order) {
  case 5: {
    Cx5 const in = reader.readTensor<Cx5>();
    Cx4 const out = ConjugateSum(in, in).sqrt();
    writer.writeTensor(HD5::Keys::Data, AddFront(out.dimensions(), 1), out.data(), reader.dimensionNames<5>());
  } break;
  case 6: {
    Cx6 temp = reader.readTensor<Cx6>();
    for (auto it = ops.cbegin(); it != ops.cend(); it++) {
      auto const op = *it;
      if (op == "rss") {
        temp = Cx6(ConjugateSum(temp, temp).sqrt().reshape(AddFront(LastN<5>(temp.dimensions()), 1)));
        continue;
      }
      it++;
      if (it == ops.cend()) {
        Log::Fail("Ops list did not make sense");
      }
      auto const fname = *it;
      HD5::Reader oread(fname);
      Cx6 const   oval = oread.readTensor<Cx6>();
      if (oval.dimensions() != temp.dimensions()) {
        Log::Fail("Mismatched dimensions {} and {}", oval.dimensions(), temp.dimensions());
      }
      if (op == "div") { temp = Cx6(temp / oval); }
    }
    writer.writeTensor(HD5::Keys::Data, temp.dimensions(), temp.data(), reader.dimensionNames<6>());
  } break;
  default: Log::Fail("Data had order {}", order);
  }
}
