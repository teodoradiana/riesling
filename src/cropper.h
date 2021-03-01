#pragma once

#include "info.h"
#include "log.h"

struct Cropper
{
  Cropper(Dims3 const &fullSz, Dims3 const &cropSz, Log &log);
  Cropper(Dims3 const &fullSz, Array3l const &cropSz, Log &log);
  Cropper(Info const &info, Dims3 const &fullSz, float const extent, bool const crop, Log &log);
  Dims3 size() const;
  Dims3 start() const;
  Cx3 newImage() const;
  Cx4 newMultichannel(long const nChan) const;
  Cx4 newSeries(long const nVols) const;
  R3 newRealImage() const;
  R4 newRealSeries(long const nVols) const;

  template <typename T>
  decltype(auto) crop3(T &&x)
  {
    return x.slice(Dims3{st_[0], st_[1], st_[2]}, Dims3{sz_[0], sz_[1], sz_[2]});
  }

  template <typename T>
  decltype(auto) crop4(T &&x)
  {
    return x.slice(Dims4{0, st_[0], st_[1], st_[2]}, Dims4{x.dimension(0), sz_[0], sz_[1], sz_[2]});
  }

private:
  Dims3 sz_, st_;
  void calcStart(Dims3 const &fullSz);
};
