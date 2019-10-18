# -*- coding: utf-8 -*-
# Copyright (C) 2017-2019 Chris N. Richardson and Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Simple mesh generation module"""

import typing

import numpy

from dolfin import cpp, fem

__all__ = ["IntervalMesh", "UnitIntervalMesh",
           "RectangleMesh", "UnitSquareMesh",
           "BoxMesh", "UnitCubeMesh"]


def IntervalMesh(comm,
                 n: int,
                 points: list,
                 ghost_mode: cpp.mesh.GhostMode = cpp.mesh.GhostMode.none) -> cpp.mesh.Mesh:
    """Create an interval mesh

    Interval mesh of the 1D line [a,b].  Given the number of cells
    (n) in the axial direction, the total number of intervals will
    be n and the total number of vertices will be (n + 1).

    Parameters
    ----------
    comm
        MPI communicator
    n
        Number of cells
    points
        Coordinates of the end points
    ghost_mode: optional
        Ghosting mode

    Returns
    -------
    mesh
        Mesh object

    Note
    ----
        Coordinate mapping is not attached

    Examples
    --------
    Create a mesh of 25 cells in the interval [-1., 1.]

    >>> from dolfin import IntervalMesh, MPI
    >>> mesh = IntervalMesh(MPI.comm_world, 25, [-1., 1.])

    Create a mesh and attach cordinate mapping

    """
    return cpp.generation.IntervalMesh.create(comm, n, points, ghost_mode)


def UnitIntervalMesh(comm,
                     n: int,
                     ghost_mode: cpp.mesh.GhostMode = cpp.mesh.GhostMode.none) -> cpp.mesh.Mesh:
    """Create a mesh on the unit interval with coordinate mapping attached

    Interval mesh of the 1D line [0,1].  Given the number of cells
    (n) in the axial direction, the total number of intervals will
    be n and the total number of vertices will be (n + 1).

    Parameters
    ----------
    comm
        MPI communicator
    n
        Number of cells
    ghost_mode: optional
        Ghosting mode

    Returns
    -------
    mesh
        Dolfin mesh object

    Examples
    --------
    Create a mesh on the unit interval with 10 cells

    >>> from dolfin import MPI, UnitIntervalMesh
    >>> mesh = UnitIntervalMesh(MPI.comm_world, 10)

    """
    mesh = IntervalMesh(comm, n, [0.0, 1.0], ghost_mode)
    mesh.geometry.coord_mapping = fem.create_coordinate_map(mesh)
    return mesh


def RectangleMesh(comm,
                  points: typing.List[numpy.array],
                  n: list,
                  cell_type: cpp.mesh.CellType = cpp.mesh.CellType.triangle,
                  ghost_mode: cpp.mesh.GhostMode = cpp.mesh.GhostMode.none,
                  diagonal: str = "right") -> cpp.mesh.Mesh:
    """Create 2D rectangle mesh spanned by two points p0 and p1

    Parameters
    ----------
    comm
        MPI communicator
    points
        List of `Points` representing vertices
    n
        List of number of cells in each direction
    cell_type: optional
        The mesh cell type: Triangle or Quadrilateral
    ghost_mode: optional
        Ghosting mode
    diagonal: optional
        Direction of diagonals: "left", "right", "left/right", "crossed"

    Notes
    ----
    Coordinate mapping is not attached.
    The parameter 'diagonal' is ignored for quadrilateral meshes.

    Examples
    --------
    Create an unit interval mesh with 25 cells

    >>> from dolfin import MPI, RectangleMesh
    >>> points = [numpy.array([0.0, 0.0, 0.0]), numpy.array([1.0, 1.0, 0.0])]
    >>> mesh = RectangleMesh(MPI.comm_world, points, [10, 10])

    """
    return cpp.generation.RectangleMesh.create(comm, points, n, cell_type, ghost_mode, diagonal)


def UnitSquareMesh(comm,
                   nx: int,
                   ny: int,
                   cell_type: cpp.mesh.CellType = cpp.mesh.CellType.tetrahedron,
                   ghost_mode: cpp.mesh.GhostMode = cpp.mesh.GhostMode.none,
                   diagonal: str = "right") -> cpp.mesh.Mesh:
    """Create a mesh of a unit square with coordinate mapping attached

    Parameters
    ----------
    comm
        MPI communicator
    nx
        Number of cells in "x" direction
    ny
        Number of cells in "y" direction
    cell_type: optional
        The mesh cell type: Triangle or Quadrilateral
    ghost_mode: optional
        Ghosting mode
    diagonal: optional
        Direction of diagonals: "left", "right", "left/right", "crossed"

    Note
    ----
    The parameter 'diagonal' is ignored for quadrilateral meshes.

    Examples
    --------
    >>> from dolfin import MPI, UnitSquareMesh
    >>> mesh = UnitSquareMesh(MPI.comm_world, 10, 10)

    """
    mesh = RectangleMesh(comm, [numpy.array([0.0, 0.0, 0.0]),
                                numpy.array([1.0, 1.0, 0.0])],
                         [nx, ny], cell_type, ghost_mode, diagonal)
    mesh.geometry.coord_mapping = fem.create_coordinate_map(mesh)
    return mesh


def BoxMesh(comm,
            points: typing.List[numpy.array],
            n: list,
            cell_type: cpp.mesh.CellType = cpp.mesh.CellType.tetrahedron,
            ghost_mode: cpp.mesh.GhostMode = cpp.mesh.GhostMode.none) -> cpp.mesh.Mesh:
    """Create box mesh

    Parameters
    ----------
    comm
        MPI communicator
    points
        List of `Points` representing vertices
    n
        List of number of cells in each direction
    cell_type: optional
        The mesh cell type: tetrahedron, hexahedron
    ghost_mode: optional
        Ghosting mode

    Notes
    ----
    Coordinate mapping is not attached.
    The parameter 'diagonal' is ignored for hexahedral meshes.

    Examples
    --------
    >>> from dolfin import MPI, BoxMesh
    >>> points = [numpy.array([0.0, 0.0, 0.0]), numpy.array([1.0, 1.0, 0.0])]
    >>> mesh = BoxMesh(MPI.comm_world, points, [5, 5, 5])

    """
    return cpp.generation.BoxMesh.create(comm, points, n, cell_type, ghost_mode)


def UnitCubeMesh(comm,
                 nx: int,
                 ny: int,
                 nz: int,
                 cell_type: cpp.mesh.CellType = cpp.mesh.CellType.tetrahedron,
                 ghost_mode: cpp.mesh.GhostMode = cpp.mesh.GhostMode.none) -> cpp.mesh.Mesh:
    """Create a mesh of a unit cube with coordinate mapping attached

    Parameters
    ----------
    comm
        MPI communicator
    nx
        Number of cells in "x" direction
    ny
        Number of cells in "y" direction
    nz
        Number of cells in "z" direction
    cell_type: optional
        The mesh cell type: tetrahedron, hexahedron
    ghost_mode: optional
        Ghosting mode

    Examples
    --------
    >>> from dolfin import MPI, UnitCubeMesh
    >>> mesh = UnitCubeMesh(MPI.comm_world, 5, 5, 5)

    """
    mesh = BoxMesh(comm, [numpy.array([0.0, 0.0, 0.0]),
                          numpy.array([1.0, 1.0, 1.0])],
                   [nx, ny, nz], cell_type, ghost_mode)
    mesh.geometry.coord_mapping = fem.create_coordinate_map(mesh)
    return mesh
