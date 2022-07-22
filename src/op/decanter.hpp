#pragma once

#include "cropper.h"
#include "fft/fft.hpp"
#include "gridBase.hpp"
#include "io/reader.hpp"
#include "threads.h"

#include <mutex>

namespace {

inline Index Reflect(Index const ii, Index const sz)
{
  if (ii < 0)
    return sz + ii;
  else if (ii >= sz)
    return ii - sz;
  else
    return ii;
}

} // namespace

namespace rl {

template <int IP, int TP>
struct Decanter final : GridBase<Cx>
{
  using typename GridBase<Cx>::Input;
  using typename GridBase<Cx>::Output;
  using FixIn = Eigen::type2index<IP>;
  using FixThrough = Eigen::type2index<TP>;

  SizedKernel<IP, TP> const *kGrid;
  Cx4 kSENSE;
  R2 basis;

  Decanter(SizedKernel<IP, TP> const *k, Mapping const &mapping, Cx4 const &kS)
    : GridBase<Cx>(mapping, 1, kS.dimension(0), mapping.frames)
    , kGrid{k}
    , kSENSE{kS}
  {
    Log::Debug(FMT_STRING("Decanter<{},{}>, dims {}"), IP, TP, this->inputDimensions());
  }

  Decanter(SizedKernel<IP, TP> const *k, Mapping const &mapping, Cx4 const &kS, R2 const b)
    : GridBase<Cx>(mapping, 1, kS.dimension(0), b.dimension(1))
    , kGrid{k}
    , kSENSE{kS}
    , basis{b}
  {
    Log::Debug(FMT_STRING("Decanter<{},{}>, dims {}"), IP, TP, this->inputDimensions());
  }

  auto inSphere(Index const isx, Index const isy, Index const isz, Sz3 const dims) const -> bool
  {
    return sqrt(
             pow((isx - (dims[0] - 1) / 2) / float((dims[0] - 1) / 2), 2) +
             pow((isy - (dims[1] - 1) / 2) / float((dims[1] - 1) / 2), 2) +
             pow((isz - (dims[2] - 1) / 2) / float((dims[2] - 1) / 2), 2)) < 1.25f;
  }

  Output A(Input const &cart) const
  {
    if (cart.dimensions() != this->inputDimensions()) {
      Log::Fail(FMT_STRING("Cartesian k-space dims {} did not match {}"), cart.dimensions(), this->inputDimensions());
    }
    Output noncart(this->outputDimensions());
    Log::Debug("Zeroing grid output");
    noncart.device(Threads::GlobalDevice()) = noncart.constant(0.f);
    auto const &idims = this->inputDimensions();
    auto const &odims = this->outputDimensions();
    Index const nC = odims[0];
    bool const hasBasis = (basis.size() > 0);
    Index const nB = hasBasis ? idims[1] : 1;
    auto const &map = this->mapping_;

    float const scale = map.scale * (hasBasis ? sqrt(basis.dimension(0)) : 1.f);

    auto grid_task = [&](Index const ibucket) {
      auto const &bucket = map.buckets[ibucket];

      Sz4 eSz = AddFront(LastN<3>(kSENSE.dimensions()), nC);
      eSz[1] += 2 * ((IP - 1) / 2);
      eSz[2] += 2 * ((IP - 1) / 2);
      eSz[3] += 2 * ((TP - 1) / 2);
      Cx4 eImg(eSz);

      for (auto ii = 0; ii < bucket.size(); ii++) {
        auto const si = bucket.indices[ii];
        auto const c = map.cart[si];
        auto const n = map.noncart[si];
        auto const ifr = hasBasis ? 0 : map.frame[si];
        Index const btp = hasBasis ? n.spoke % basis.dimension(0) : 0;
        R3 const k = this->kGrid->k(map.offset[si]) * scale;
        eImg.setZero();
        for (Index isz = 0; isz < kSENSE.dimension(3); isz++) {
          for (Index isy = 0; isy < kSENSE.dimension(2); isy++) {
            for (Index isx = 0; isx < kSENSE.dimension(1); isx++) {
              if (inSphere(isx, isy, isz, LastN<3>(kSENSE.dimensions()))) {
                for (Index iz = 0; iz < TP; iz++) {
                  Index const iiz = iz + isz;
                  for (Index iy = 0; iy < IP; iy++) {
                    Index const iiy = iy + isy;
                    for (Index ix = 0; ix < IP; ix++) {
                      Index const iix = ix + isx;
                      float const kval = k(IP - 1 - ix, IP - 1 - iy, TP - 1 - iz);
                      for (Index ic = 0; ic < nC; ic++) {
                        eImg(ic, iix, iiy, iiz) += kval * kSENSE(ic, isx, isy, isz);
                      }
                    }
                  }
                }
              }
            }
          }
        }

        Index const stX = c.x - (eSz[0] - 1) / 2;
        Index const stY = c.y - (eSz[1] - 1) / 2;
        Index const stZ = c.z - (eSz[2] - 1) / 2;

        for (Index iz = 0; iz < eSz[2]; iz++) {
          Index const iiz = stZ + iz;
          for (Index iy = 0; iy < eSz[1]; iy++) {
            Index const iiy = stY + iy;
            for (Index ix = 0; ix < eSz[0]; ix++) {
              Index const iix = stX + ix;
              if (inSphere(ix, iy, iz, LastN<3>(eSz))) {
                Cx bval = 0;
                for (Index ib = 0; ib < nB; ib++) {
                  bval += (hasBasis ? basis(btp, ib) : 1.f) * cart(0, ib + ifr, iix, iiy, iiz);
                }
                for (Index ic = 0; ic < nC; ic++) {
                  noncart(ic, n.read, n.spoke) += bval * eImg(ic, ix, iy, iz);
                }
              }
            }
          }
        }
      }
    };

    Threads::For(grid_task, map.buckets.size(), "Grid Forward");
    return noncart;
  }

  Input &Adj(Output const &noncart) const
  {
    Log::Debug("Grid Adjoint");
    if (noncart.dimensions() != this->outputDimensions()) {
      Log::Fail(
        FMT_STRING("Noncartesian k-space dims {} did not match {}"), noncart.dimensions(), this->outputDimensions());
    }
    auto const &idims = this->inputDimensions();
    auto const &odims = this->outputDimensions();
    auto const &map = this->mapping_;
    Index const nC = odims[0];
    Index const nFr = idims[1];
    bool const hasBasis = (basis.size() > 0);
    float const scale = map.scale * (hasBasis ? sqrt(basis.dimension(0)) : 1.f);

    std::mutex writeMutex;
    auto grid_task = [&](Index ibucket) {
      auto const &bucket = map.buckets[ibucket];
      auto bSz = bucket.gridSize();
      bSz[0] += kSENSE.dimension(1);
      bSz[1] += kSENSE.dimension(2);
      bSz[2] += kSENSE.dimension(3);
      auto minCorner = bucket.minCorner;
      minCorner[0] -= (kSENSE.dimension(1) - 1) / 2;
      minCorner[1] -= (kSENSE.dimension(2) - 1) / 2;
      minCorner[2] -= (kSENSE.dimension(3) - 1) / 2;
      Cx4 out(AddFront(bSz, nFr));
      out.setZero();
      Cx1 combined(hasBasis ? nFr : 1);
      for (auto ii = 0; ii < bucket.size(); ii++) {
        auto const si = bucket.indices[ii];
        auto const c = map.cart[si];
        auto const n = map.noncart[si];
        auto const fr = map.frame[si];
        auto const frscale = scale * (this->weightFrames_ ? map.frameWeights[fr] : 1.f);
        typename SizedKernel<IP, TP>::KTensor const k = this->kGrid->k(map.offset[si]) * frscale;
        Index const btp = hasBasis ? n.spoke % basis.dimension(0) : 0;

        Index const stSX = c.x - ((kSENSE.dimension(1) - 1) / 2) - minCorner[0];
        Index const stSY = c.y - ((kSENSE.dimension(2) - 1) / 2) - minCorner[1];
        Index const stSZ = c.z - ((kSENSE.dimension(3) - 1) / 2) - minCorner[2];
        Cx1 const sample = noncart.chip(n.spoke, 2).chip(n.read, 1);

        for (Index isz = 0; isz < kSENSE.dimension(3); isz++) {
          Index const rsz = kSENSE.dimension(3) - 1 - isz;
          Index const stZ = stSZ + isz - ((TP - 1) / 2);
          for (Index isy = 0; isy < kSENSE.dimension(2); isy++) {
            Index const rsy = kSENSE.dimension(2) - 1 - isy;
            Index const stY = stSY + isy - ((IP - 1) / 2);
            for (Index isx = 0; isx < kSENSE.dimension(1); isx++) {
              Index const rsx = kSENSE.dimension(1) - 1 - isx;
              Index const stX = stSX + isx - ((IP - 1) / 2);
              if (inSphere(isx, isy, isz, LastN<3>(kSENSE.dimensions()))) {
                combined.setZero();
                if (hasBasis) {
                  for (Index ifr = 0; ifr < nFr; ifr++) {
                    for (Index ic = 0; ic < nC; ic++) {
                      combined(ifr) +=
                        basis(btp, ifr) * noncart(ic, n.read, n.spoke) * std::conj(kSENSE(ic, rsx, rsy, rsz));
                    }
                  }
                } else {
                  for (Index ic = 0; ic < nC; ic++) {
                    combined(0) += noncart(ic, n.read, n.spoke) * std::conj(kSENSE(ic, rsx, rsy, rsz));
                  }
                }

                for (Index iz = 0; iz < TP; iz++) {
                  Index const iiz = stZ + iz;
                  for (Index iy = 0; iy < IP; iy++) {
                    Index const iiy = stY + iy;
                    for (Index ix = 0; ix < IP; ix++) {
                      Index const iix = stX + ix;
                      float const kval = k(ix, iy, iz);
                      if (hasBasis) {
                        for (Index ib = 0; ib < nFr; ib++) {
                          out(ib, iix, iiy, iiz) += combined(ib) * kval;
                        }
                      } else {
                        out(fr, iix, iiy, iiz) += combined(0) * kval;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      {
        std::scoped_lock lock(writeMutex);
        for (Index iz = 0; iz < bSz[2]; iz++) {
          Index const iiz = Reflect(minCorner[2] + iz, idims[4]);
          for (Index iy = 0; iy < bSz[1]; iy++) {
            Index const iiy = Reflect(minCorner[1] + iy, idims[3]);
            for (Index ix = 0; ix < bSz[0]; ix++) {
              Index const iix = Reflect(minCorner[0] + ix, idims[2]);
              for (Index ifr = 0; ifr < nFr; ifr++) {
                this->ws_->operator()(0, ifr, iix, iiy, iiz) += out(ifr, ix, iy, iz);
              }
            }
          }
        }
      }
    };

    Log::Debug("Zeroing workspace");
    this->ws_->device(Threads::GlobalDevice()) = this->ws_->constant(0.f);
    Threads::For(grid_task, map.buckets.size(), "Grid Adjoint");
    Log::Debug("Grid Adjoint finished");
    return *(this->ws_);
  }

  R3 apodization(Sz3 const sz) const
  {
    auto gridSz = this->mapping().cartDims;
    Cx3 temp(gridSz);
    auto const fft = FFT::Make<3, 3>(gridSz);
    temp.setZero();
    auto const k = kGrid->k(Point3{0, 0, 0});
    Crop3(temp, k.dimensions()) = k.template cast<Cx>();
    Log::Tensor(temp, "apo-kGrid");
    fft->reverse(temp);
    R3 a = Crop3(R3(temp.real()), sz);
    float const scale = sqrt(Product(gridSz));
    Log::Print(FMT_STRING("Apodization size {} scale factor: {}"), fmt::join(a.dimensions(), ","), scale);
    a.device(Threads::GlobalDevice()) = a * a.constant(scale);
    Log::Tensor(a, "apo-final");
    return a;
  }
};

inline std::unique_ptr<GridBase<Cx>>
make_decanter(Kernel const *k, Mapping const &m, Cx4 const &kS, std::string const &basisFile)
{
  if (!basisFile.empty()) {
    HD5::Reader basisReader(basisFile);
    R2 const b = basisReader.readTensor<R2>(HD5::Keys::Basis);
    switch (k->inPlane()) {
    case 1:
      return std::make_unique<Decanter<1, 1>>(dynamic_cast<SizedKernel<1, 1> const *>(k), m, kS, b);
    case 3:
      if (k->throughPlane() == 1) {
        return std::make_unique<Decanter<3, 1>>(dynamic_cast<SizedKernel<3, 1> const *>(k), m, kS, b);
      } else if (k->throughPlane() == 3) {
        return std::make_unique<Decanter<3, 3>>(dynamic_cast<SizedKernel<3, 3> const *>(k), m, kS, b);
      }
      break;
    case 5:
      if (k->throughPlane() == 1) {
        return std::make_unique<Decanter<5, 1>>(dynamic_cast<SizedKernel<5, 1> const *>(k), m, kS, b);
      } else if (k->throughPlane() == 5) {
        return std::make_unique<Decanter<5, 5>>(dynamic_cast<SizedKernel<5, 5> const *>(k), m, kS, b);
      }
      break;
    }
  } else {
    switch (k->inPlane()) {
    case 1:
      return std::make_unique<Decanter<1, 1>>(dynamic_cast<SizedKernel<1, 1> const *>(k), m, kS);
    case 3:
      if (k->throughPlane() == 1) {
        return std::make_unique<Decanter<3, 1>>(dynamic_cast<SizedKernel<3, 1> const *>(k), m, kS);
      } else if (k->throughPlane() == 3) {
        return std::make_unique<Decanter<3, 3>>(dynamic_cast<SizedKernel<3, 3> const *>(k), m, kS);
      }
      break;
    case 5:
      if (k->throughPlane() == 1) {
        return std::make_unique<Decanter<5, 1>>(dynamic_cast<SizedKernel<5, 1> const *>(k), m, kS);
      } else if (k->throughPlane() == 5) {
        return std::make_unique<Decanter<5, 5>>(dynamic_cast<SizedKernel<5, 5> const *>(k), m, kS);
      }
      break;
    }
  }
  Log::Fail(FMT_STRING("No grids implemented for in-plane kernel width {}"), k->inPlane());
}

} // namespace rl
