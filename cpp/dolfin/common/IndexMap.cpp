// Copyright (C) 2015 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "IndexMap.h"
#include <algorithm>
#include <map>
#include <set>
#include <vector>

using namespace dolfin;
using namespace dolfin::common;

//-----------------------------------------------------------------------------
IndexMap::IndexMap(MPI_Comm mpi_comm, std::int32_t local_size,
                   const std::vector<std::int64_t>& ghosts,
                   std::size_t block_size)
    : block_size(block_size), _mpi_comm(mpi_comm), _myrank(MPI::rank(mpi_comm)),
      _ghosts(ghosts.size()), _ghost_owners(ghosts.size())
{
  // Calculate offsets
  MPI::all_gather(_mpi_comm, (std::int64_t)local_size, _all_ranges);

  const std::int32_t mpi_size = dolfin::MPI::size(_mpi_comm);
  for (std::int32_t i = 1; i < mpi_size; ++i)
    _all_ranges[i] += _all_ranges[i - 1];

  _all_ranges.insert(_all_ranges.begin(), 0);

  std::vector<std::vector<std::int32_t>> send_index(mpi_size);
  _remotes.resize(mpi_size);

  for (std::size_t i = 0; i < ghosts.size(); ++i)
  {
    _ghosts[i] = ghosts[i];
    _ghost_owners[i] = owner(ghosts[i]);
    assert(_ghost_owners[i] != _myrank);

    // Send desired index to remote
    const int p = _ghost_owners[i];
    send_index[p].push_back(_ghosts[i] - _all_ranges[p]);
  }

  // Use alltoall (global)
  MPI::all_to_all(_mpi_comm, send_index, _remotes);

  // ------------------------------------------------------------
  // Now repeat using neighbourhood
  // ------------------------------------------------------------

  // Find all neighbour counts, both send and receive
  std::vector<std::int32_t> send_nghosts(mpi_size, 0);

  for (std::size_t i = 0; i < ghosts.size(); ++i)
  {
    _ghosts[i] = ghosts[i];
    _ghost_owners[i] = owner(ghosts[i]);
    assert(_ghost_owners[i] != _myrank);

    // Send desired index to remote
    const int p = _ghost_owners[i];
    ++send_nghosts[p];
  }

  std::vector<std::int32_t> recv_nghosts(mpi_size);
  MPI_Alltoall(send_nghosts.data(), 1, MPI_INT, recv_nghosts.data(), 1, MPI_INT,
               _mpi_comm);

  std::vector<std::int32_t> neighbours;
  std::map<std::int32_t, std::int32_t> proc_to_nbr;
  std::vector<std::int32_t> send_sizes;
  std::vector<std::int32_t> send_offsets = {0};
  std::vector<std::int32_t> recv_sizes;
  std::vector<std::int32_t> recv_offsets = {0};
  for (std::int32_t p = 0; p < mpi_size; ++p)
    if (send_nghosts[p] > 0 or recv_nghosts[p] > 0)
    {
      proc_to_nbr.insert({p, (int)neighbours.size()});
      neighbours.push_back(p);
      send_sizes.push_back(send_nghosts[p]);
      send_offsets.push_back(send_offsets.back() + send_nghosts[p]);
      recv_sizes.push_back(recv_nghosts[p]);
      recv_offsets.push_back(recv_offsets.back() + recv_nghosts[p]);
    }

  // No communication is needed to build the graph with complete adjacent
  // information
  MPI_Dist_graph_create_adjacent(
      _mpi_comm, neighbours.size(), neighbours.data(), MPI_UNWEIGHTED,
      neighbours.size(), neighbours.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
      false, &_neighbour_comm);

  std::int32_t num_neighbours = neighbours.size();
  std::vector<int> sources(num_neighbours);
  std::vector<int> dests(num_neighbours);
  MPI_Dist_graph_neighbors(_neighbour_comm, num_neighbours, sources.data(),
                           NULL, num_neighbours, dests.data(), NULL);

  assert(sources == dests);

  int nbr_size = MPI::size(_neighbour_comm);
  std::stringstream s;
  s << _myrank << "/" << nbr_size << ": ";
  for (int q : sources)
    s << q << " ";
  s << "\n";
  std::cout << s.str();

  std::vector<std::int32_t> nbr_send_index(send_offsets.back());
  std::vector<std::int32_t> nbr_recv_index(recv_offsets.back());
  std::vector<std::int32_t> count(send_offsets);
  for (std::size_t i = 0; i < ghosts.size(); ++i)
  {
    const int p = _ghost_owners[i];
    const int np = proc_to_nbr[p];
    nbr_send_index[count[np]] = _ghosts[i] - _all_ranges[p];
    ++count[np];
  }

  MPI_Neighbor_alltoallv(nbr_send_index.data(), send_sizes.data(),
                         send_offsets.data(), MPI_INT, nbr_recv_index.data(),
                         recv_sizes.data(), recv_offsets.data(), MPI_INT,
                         _neighbour_comm);

  s.str("");

  s << "RANK = " << _myrank << "\n----------------\n";
  for (auto& q : nbr_send_index)
    s << q << " ";
  s << "\n---------------\n";
  for (auto& q : nbr_recv_index)
    s << q << " ";
  s << "\n---------------";

  std::cout << s.str() << "\n";
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> IndexMap::local_range() const
{
  return {{_all_ranges[_myrank], _all_ranges[_myrank + 1]}};
}
//-----------------------------------------------------------------------------
std::int32_t IndexMap::num_ghosts() const { return _ghosts.size(); }
//-----------------------------------------------------------------------------
std::int32_t IndexMap::size_local() const
{
  return _all_ranges[_myrank + 1] - _all_ranges[_myrank];
}
//-----------------------------------------------------------------------------
std::int64_t IndexMap::size_global() const { return _all_ranges.back(); }
//-----------------------------------------------------------------------------
const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& IndexMap::ghosts() const
{
  return _ghosts;
}
//-----------------------------------------------------------------------------
int IndexMap::owner(std::int64_t global_index) const
{
  auto it
      = std::upper_bound(_all_ranges.begin(), _all_ranges.end(), global_index);
  return std::distance(_all_ranges.begin(), it) - 1;
}
//-----------------------------------------------------------------------------
const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>&
IndexMap::ghost_owners() const
{
  return _ghost_owners;
}
//----------------------------------------------------------------------------
Eigen::Array<std::int64_t, Eigen::Dynamic, 1>
IndexMap::indices(bool unroll_block) const
{
  const int bs = unroll_block ? this->block_size : 1;
  const std::array<std::int64_t, 2> local_range = this->local_range();
  const std::int32_t size_local = this->size_local() * bs;

  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> indx(size_local
                                                     + num_ghosts() * bs);
  std::iota(indx.data(), indx.data() + size_local, bs * local_range[0]);
  for (Eigen::Index i = 0; i < num_ghosts(); ++i)
  {
    for (Eigen::Index j = 0; j < bs; ++j)
      indx[size_local + bs * i + j] = bs * _ghosts[i] + j;
  }

  return indx;
}
//----------------------------------------------------------------------------
MPI_Comm IndexMap::mpi_comm() const { return _mpi_comm; }
//----------------------------------------------------------------------------
void IndexMap::scatter_fwd(const std::vector<std::int64_t>& local_data,
                           std::vector<std::int64_t>& remote_data, int n) const
{
  scatter_fwd_impl(local_data, remote_data, n);
}
//-----------------------------------------------------------------------------
void IndexMap::scatter_fwd(const std::vector<std::int32_t>& local_data,
                           std::vector<std::int32_t>& remote_data, int n) const
{
  scatter_fwd_impl(local_data, remote_data, n);
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t>
IndexMap::scatter_fwd(const std::vector<std::int64_t>& local_data, int n) const
{
  std::vector<std::int64_t> remote_data;
  scatter_fwd_impl(local_data, remote_data, n);
  return remote_data;
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t>
IndexMap::scatter_fwd(const std::vector<std::int32_t>& local_data, int n) const
{
  std::vector<std::int32_t> remote_data;
  scatter_fwd_impl(local_data, remote_data, n);
  return remote_data;
}
//-----------------------------------------------------------------------------
void IndexMap::scatter_rev(std::vector<std::int64_t>& local_data,
                           const std::vector<std::int64_t>& remote_data, int n,
                           MPI_Op op) const
{
  scatter_rev_impl(local_data, remote_data, n, op);
}
//-----------------------------------------------------------------------------
void IndexMap::scatter_rev(std::vector<std::int32_t>& local_data,
                           const std::vector<std::int32_t>& remote_data, int n,
                           MPI_Op op) const
{
  scatter_rev_impl(local_data, remote_data, n, op);
}
//-----------------------------------------------------------------------------
template <typename T>
void IndexMap::scatter_fwd_impl(const std::vector<T>& local_data,
                                std::vector<T>& remote_data, int n) const
{
  const std::size_t _size_local = size_local();
  assert(local_data.size() == n * _size_local);
  remote_data.resize(n * num_ghosts());

  int mpi_size = MPI::size(_mpi_comm);
  std::vector<T> send_data;
  std::vector<int> send_data_offsets = {0};
  std::vector<int> send_data_sizes;

  std::vector<T> recv_data(_ghosts.size() * n);

  for (int p = 0; p < mpi_size; ++p)
  {
    const std::vector<std::int32_t>& rp = _remotes[p];

    for (std::size_t i = 0; i < rp.size(); ++i)
    {
      send_data.insert(send_data.end(), local_data.begin() + rp[i] * n,
                       local_data.begin() + rp[i] * n + n);
    }
    send_data_sizes.push_back(send_data.size() - send_data_offsets.back());
    send_data_offsets.push_back(send_data.size());
  }

  std::vector<int> recv_data_sizes(mpi_size, 0);
  for (int i = 0; i < _ghosts.size(); ++i)
    recv_data_sizes[_ghost_owners[i]] += n;
  std::vector<int> recv_data_offsets = {0};
  for (int q : recv_data_sizes)
    recv_data_offsets.push_back(recv_data_offsets.back() + q);

  MPI_Alltoallv(send_data.data(), send_data_sizes.data(),
                send_data_offsets.data(), MPI::mpi_type<T>(), recv_data.data(),
                recv_data_sizes.data(), recv_data_offsets.data(),
                MPI::mpi_type<T>(), _mpi_comm);

  for (int i = 0; i < _ghosts.size(); ++i)
  {
    int p = _ghost_owners[i];
    std::copy(recv_data.begin() + recv_data_offsets[p],
              recv_data.begin() + recv_data_offsets[p] + n,
              remote_data.begin() + i * n);
    recv_data_offsets[p] += n;
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void IndexMap::scatter_rev_impl(std::vector<T>& local_data,
                                const std::vector<T>& remote_data, int n,
                                MPI_Op op) const
{
  assert((std::int32_t)remote_data.size() == n * num_ghosts());
  local_data.resize(n * size_local(), 0);

  int mpi_size = MPI::size(_mpi_comm);

  std::vector<int> send_data_sizes(mpi_size, 0);
  for (int i = 0; i < _ghosts.size(); ++i)
    send_data_sizes[_ghost_owners[i]] += n;
  std::vector<int> send_data_offsets = {0};
  for (int q : send_data_sizes)
    send_data_offsets.push_back(send_data_offsets.back() + q);
  std::vector<T> send_data(send_data_offsets.back());

  // Copy offsets
  std::vector<int> count(send_data_offsets);
  for (int i = 0; i < _ghosts.size(); ++i)
  {
    int p = _ghost_owners[i];
    std::copy(remote_data.begin() + i * n, remote_data.begin() + i * n + n,
              send_data.begin() + count[p]);
    count[p] += n;
  }

  std::vector<int> recv_data_offsets = {0};
  std::vector<int> recv_data_sizes;
  for (int p = 0; p < mpi_size; ++p)
  {
    int rpsize = _remotes[p].size() * n;
    recv_data_sizes.push_back(rpsize);
    recv_data_offsets.push_back(recv_data_offsets.back() + rpsize);
  }
  std::vector<T> recv_data(recv_data_offsets.back());

  MPI_Alltoallv(send_data.data(), send_data_sizes.data(),
                send_data_offsets.data(), MPI::mpi_type<T>(), recv_data.data(),
                recv_data_sizes.data(), recv_data_offsets.data(),
                MPI::mpi_type<T>(), _mpi_comm);

  for (int p = 0; p < mpi_size; ++p)
  {
    const std::vector<std::int32_t>& rp = _remotes[p];
    assert(recv_data_sizes[p] == (int)rp.size() * n);

    for (std::size_t i = 0; i < rp.size(); ++i)
    {
      for (int j = 0; j < n; ++j)
        local_data[rp[i] * n + j] = recv_data[recv_data_offsets[p] + i * n + j];
    }
  }
}
//-----------------------------------------------------------------------------
