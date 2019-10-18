// Copyright (C) 2005-2017 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/Mesh.h>

namespace dolfin
{

namespace generation
{

/// Mesh of the 3D rectangular prism spanned by two points p0 and p1.
/// Given the number of cells (nx, ny, nz) in each direction, the total number
/// of cells will be 6*nx*ny*nz for tetrahedral meshes and nx*ny*nz for
/// hexahedral meshes.  The total number of vertices will be
/// (nx + 1)*(ny + 1)*(nz + 1) for both tetrahedral and hexahedral meshes.

class BoxMesh
{
public:
  /// Create a uniform finite element _Mesh_ over the rectangular prism
  /// spanned by the two points p0 and p1. The order of the
  /// two points is not important in terms of minimum and maximum
  /// coordinates.
  ///
  /// @param[in] comm MPI communicator to build mesh on
  /// @param[in] p Points of box
  /// @param[in] n Number of cells in each direction.
  /// @param[in] cell_type Tetrahedron or hexahedron
  /// @param[in] ghost_mode Mesh ghosting mode
  /// @return Mesh
  static mesh::Mesh create(MPI_Comm comm,
                           const std::array<Eigen::Vector3d, 2>& p,
                           std::array<std::size_t, 3> n,
                           mesh::CellType cell_type,
                           const mesh::GhostMode ghost_mode);
};
} // namespace generation
} // namespace dolfin
