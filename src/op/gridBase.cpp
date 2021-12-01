#include "gridBase.h"

GridBase::GridBase(Mapping map, bool const unsafe, Log &log)
  : mapping_{std::move(map)}
  , safe_{!unsafe}
  , sdcPow_{1.f}
  , log_{log}
{
}

void GridBase::setSDC(float const d)
{
  std::fill(mapping_.sdc.begin(), mapping_.sdc.end(), d);
}

void GridBase::setSDC(R2 const &sdc)
{
  std::transform(
    mapping_.noncart.begin(),
    mapping_.noncart.end(),
    mapping_.sdc.begin(),
    [&sdc](NoncartesianIndex const &nc) { return sdc(nc.read, nc.spoke); });
}

R2 GridBase::SDC() const
{
  R2 sdc(mapping_.noncartDims[1], mapping_.noncartDims[2]);
  sdc.setZero();
  for (size_t ii = 0; ii < mapping_.noncart.size(); ii++) {
    sdc(mapping_.noncart[ii].read, mapping_.noncart[ii].spoke) = mapping_.sdc[ii];
  }
  return sdc;
}

void GridBase::setSDCPower(float const p)
{
  sdcPow_ = p;
}

void GridBase::setUnsafe()
{
  safe_ = true;
}

void GridBase::setSafe()
{
  safe_ = false;
}

Mapping const &GridBase::mapping() const
{
  return mapping_;
}
