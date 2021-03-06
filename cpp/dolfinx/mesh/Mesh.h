// Copyright (C) 2006-2020 Anders Logg, Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Geometry.h"
#include "Topology.h"
#include "cell_types.h"
#include "utils.h"
#include <Eigen/Dense>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/UniqueIdGenerator.h>
#include <string>
#include <utility>

namespace dolfinx
{

namespace fem
{
class CoordinateElement;
}

namespace graph
{
template <typename T>
class AdjacencyList;
}

namespace mesh
{

/// Enum for different partitioning ghost modes
enum class GhostMode : int
{
  none,
  shared_facet,
  shared_vertex
};

/// A Mesh consists of a set of connected and numbered mesh topological
/// entities, and geometry data

class Mesh
{
public:
  /// Create a mesh
  /// @param[in] comm MPI Communicator
  /// @param[in] topology Mesh topology
  /// @param[in] geometry Mesh geometry
  template <typename Topology, typename Geometry>
  Mesh(MPI_Comm comm, Topology&& topology, Geometry&& geometry)
      : _topology(std::forward<Topology>(topology)),
        _geometry(std::forward<Geometry>(geometry)), _mpi_comm(comm)
  {
    // Do nothing
  }

  /// Copy constructor
  /// @param[in] mesh Mesh to be copied
  Mesh(const Mesh& mesh) = default;

  /// Move constructor
  /// @param mesh Mesh to be moved.
  Mesh(Mesh&& mesh) = default;

  /// Destructor
  ~Mesh() = default;

  // Assignment operator
  Mesh& operator=(const Mesh& mesh) = delete;

  /// Assignment move operator
  /// @param mesh Another Mesh object
  Mesh& operator=(Mesh&& mesh) = default;

  // TODO: Is there any use for this? In many situations one has to get the
  // topology of a const Mesh, which is done by Mesh::topology_mutable. Note
  // that the python interface (calls Mesh::topology()) may still rely on it.
  /// Get mesh topology
  /// @return The topology object associated with the mesh.
  Topology& topology();

  /// Get mesh topology (const version)
  /// @return The topology object associated with the mesh.
  const Topology& topology() const;

  /// Get mesh topology if one really needs the mutable version
  /// @return The topology object associated with the mesh.
  Topology& topology_mutable() const;

  /// Get mesh geometry
  /// @return The geometry object associated with the mesh
  Geometry& geometry();

  /// Get mesh geometry (const version)
  /// @return The geometry object associated with the mesh
  const Geometry& geometry() const;

  /// Compute minimum cell size in mesh, measured greatest distance
  /// between any two vertices of a cell.
  /// @return The minimum cell size. The size is computed using
  ///         Cell::h()
  double hmin() const;

  /// Compute maximum cell size in mesh, measured greatest distance
  /// between any two vertices of a cell
  /// @return The maximum cell size. The size is computed using
  ///         Cell::h()
  double hmax() const;

  /// Compute minimum cell inradius
  /// @return double The minimum of cells' inscribed sphere radii
  double rmin() const;

  /// Compute maximum cell inradius
  /// @return The maximum of cells' inscribed sphere radii
  double rmax() const;

  /// Compute hash of mesh, currently based on the has of the mesh
  /// geometry and mesh topology
  /// @return A tree-hashed value of the coordinates over all MPI
  ///         processes
  std::size_t hash() const;

  /// Get unique identifier for the mesh
  /// @returns The unique identifier associated with the object
  std::size_t id() const { return _unique_id; }

  /// Mesh MPI communicator
  /// @return The communicator on which the mesh is distributed
  MPI_Comm mpi_comm() const;

  /// Name
  std::string name = "mesh";

private:
  // Mesh topology:
  // TODO: This is mutable because of the current memory management within
  // mesh::Topology. It allows to obtain a non-const Topology from a
  // const mesh (via Mesh::topology_mutable()).
  //
  mutable Topology _topology;

  // Mesh geometry
  Geometry _geometry;

  // MPI communicator
  dolfinx::MPI::Comm _mpi_comm;

  // Unique identifier
  std::size_t _unique_id = common::UniqueIdGenerator::id();
};

/// Create a mesh
Mesh create_mesh(
    MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& cells,
    const fem::CoordinateElement& element,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        x,
    GhostMode ghost_mode,
    std::function<const graph::AdjacencyList<std::int32_t>(
        MPI_Comm, int, const mesh::CellType,
        const graph::AdjacencyList<std::int64_t>&, mesh::GhostMode)>
        partitioner
    = &mesh::partition_cells);

} // namespace mesh
} // namespace dolfinx
