// Copyright (C) 2013 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/mesh/MeshFunction.h>
#include <map>
#include <utility>
#include <vector>

namespace dolfin
{
namespace mesh
{
class Mesh;

/// This class computes map from slave entity to master entity

class PeriodicBoundaryComputation
{
public:
  /// For entities of dimension dim, compute map from a slave entity
  /// on this process (local index) to its master entity (owning
  /// process, local index on owner). If a master entity is shared
  /// by processes, only one of the owning processes is returned.
  static std::map<std::int32_t, std::pair<std::int32_t, std::int32_t>>
  compute_periodic_pairs(
      const Mesh& mesh,
      const std::function<Eigen::Array<bool, Eigen::Dynamic, 1>(
          const Eigen::Ref<
              const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>> x)>& mark,
      const std::size_t dim, const double tol);

  /// This function returns a MeshFunction which marks mesh entities
  /// of dimension dim according to:
  ///
  ///     2: slave entities
  ///     1: master entities
  ///     0: all other entities
  ///
  /// It is useful for visualising and debugging the Expression::map
  /// function that is used to apply periodic boundary conditions.
  static MeshFunction<std::size_t> masters_slaves(
      std::shared_ptr<const Mesh> mesh,
      const std::function<Eigen::Array<bool, Eigen::Dynamic, 1>(
          const Eigen::Ref<
              const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>> x)>& mark,
      const std::size_t dim, const double tol);
};
} // namespace mesh
} // namespace dolfin
