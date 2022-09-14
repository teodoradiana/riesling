#pragma once

#include "io/hd5-core.hpp"

#include "info.hpp"
#include "trajectory.hpp"

#include <map>
#include <string>

namespace rl {

namespace HD5 {

/*
 * This class is for reading tensors out of generic HDF5 files. Used for SDC, SENSE maps, etc.
 */
struct Reader
{
  Reader(Reader const &) = delete;
  Reader(std::string const &fname);
  ~Reader();

  auto list() -> std::vector<std::string>; // List all datasets
  auto rank(std::string const &label) -> Index; // Determine rank of a dataset
  template <int Rank> auto dimensions(std::string const &label) -> Eigen::DSizes<Index, Rank>; // Get Tensor dimensions

  template <typename T>
  T readTensor(std::string const &label);
  template <typename T>
  void readTensor(std::string const &label, T &tensor); // Read with size checks
  template <typename Derived>
  Derived readMatrix(std::string const &label);

  Info readInfo(); // Read the info struct from a file

protected:
  Handle handle_;
};

/*
 * This class is for reading "full" riesling datasets - i.e. the HDF5 file must contain the Info
 * struct, a Trajectory, and then some valid data
 */
struct RieslingReader : Reader
{
  RieslingReader(std::string const &fname);
  std::map<std::string, float> readMeta() const;
  Trajectory const &trajectory() const;
  Cx3 const &noncartesian(Index const index); // This will be cached

private:
  Index currentNCVol_;
  Trajectory traj_;
  Cx3 nc_;
};

} // namespace HD5
}
