// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

/*
tensors:
1. act  (A): input feature map
2. gate (G): B matrix for first gemm, output will do activation(Silu)
3. up   (U): B matrix for first gemm
4. down (D): B matrix for second gemm
                                                                  N1
                                                                 /   \
                                                                 +----------+ |
                                                                 |   Down   | |
                                                                 x----------x |
                        hidden               hidden           K1 |          | |
                          N0                   N0                x----------x |
             |   +------x-----x------+------x-----x------+       |          | |
    dim      |   | Gate |     |      | Up   |     |      |       |          | |
  contiguous |   |      |     |      |      |     |      |       |          | |
             |   |      |     |      |      |     |      |       |          | |
             v   +------x-----x------+------x-----x------+       +----------+ V
      K0                |     |             |     |                    | contiguous
     /  \               v     v             v     v                    |
    +---------+  +------x-----x------+------x-----x------+             |
M0  |    A    |  |      |     |      |      |     |      |             |
    +---------+  +------x-----x------+------x-----x------+             |
    ---------->           |                    |                       |
    contiguous            |                    V                       V
                          |                 x-----x              +----------+
                          +------------> M1 |  Y  |  --------->  |  Out(O)  |
                             ACT            x-----x              +----------+
                                              K1 = N0                 dim

* Note: Act could be Gelu/Silu/...
* Note: some model does not have Up
*/
template <typename BlockTile_0_,
          typename WarpPerBlock_0_,
          typename WarpTile_0_,
          typename BlockTile_1_,
          typename WarpPerBlock_1_,
          typename WarpTile_1_>
struct FusedMoeGemmShape
{
    using BlockTile_0    = remove_cvref_t<BlockTile_0_>; // S<32, 512, 128>
    using WarpPerBlock_0 = remove_cvref_t<WarpPerBlock_0_>; // S<1, 4, 1>
    using WarpTile_0     = remove_cvref_t<WarpTile_0_>; // S<16, 16, 32>
    using BlockTile_1    = remove_cvref_t<BlockTile_1_>; // S<32, 128, 512>
    using WarpPerBlock_1 = remove_cvref_t<WarpPerBlock_1_>; // S<1, 4, 1>
    using WarpTile_1     = remove_cvref_t<WarpTile_1_>; // S<16, 16, 32>

    static constexpr index_t NumWarps =
        reduce_on_sequence(WarpPerBlock_0{}, multiplies{}, number<1>{}); // 1x4x1=4

    // TODO: we don't support half warps aound to 1 warp here
    static_assert(NumWarps == reduce_on_sequence(WarpPerBlock_1{}, multiplies{}, number<1>{})); // 1x4x1=4

    static constexpr index_t Block_M0        = BlockTile_0::at(number<0>{}); // 32
    static constexpr index_t Block_N0        = BlockTile_0::at(number<1>{}); // 512
    static constexpr index_t Block_K0        = BlockTile_0::at(number<2>{}); // 128
    static constexpr index_t WarpPerBlock_M0 = WarpPerBlock_0::at(number<0>{}); // 1
    static constexpr index_t WarpPerBlock_N0 = WarpPerBlock_0::at(number<1>{}); // 4
    static constexpr index_t WarpPerBlock_K0 = WarpPerBlock_0::at(number<2>{}); // 1
    static constexpr index_t Warp_M0         = WarpTile_0::at(number<0>{}); // 16
    static constexpr index_t Warp_N0         = WarpTile_0::at(number<1>{}); // 16
    static constexpr index_t Warp_K0         = WarpTile_0::at(number<2>{}); // 32

    static constexpr index_t ThreadPerBlock_M0 = Warp_M0 * WarpPerBlock_M0; // 16x1=16
    static constexpr index_t ThreadPerBlock_N0 = Warp_N0 * WarpPerBlock_N0; // 16x4=64
    static constexpr index_t ThreadPerBlock_K0 = Warp_K0 * WarpPerBlock_K0; // 32x1=32
    static_assert(Block_M0 % ThreadPerBlock_M0 == 0);
    static_assert(Block_N0 % ThreadPerBlock_N0 == 0);
    static_assert(Block_K0 % ThreadPerBlock_K0 == 0);
    static constexpr index_t Repeat_M0 = Block_M0 / ThreadPerBlock_M0; // 32/16=2
    static constexpr index_t Repeat_N0 = Block_N0 / ThreadPerBlock_N0; // 512/64=8
    static constexpr index_t Repeat_K0 = Block_K0 / ThreadPerBlock_K0; // 128/32=4

    static constexpr index_t Block_M1        = BlockTile_1::at(number<0>{}); // 32
    static constexpr index_t Block_N1        = BlockTile_1::at(number<1>{}); // 128
    static constexpr index_t Block_K1        = BlockTile_1::at(number<2>{}); // 512
    static constexpr index_t WarpPerBlock_M1 = WarpPerBlock_1::at(number<0>{}); // 1
    static constexpr index_t WarpPerBlock_N1 = WarpPerBlock_1::at(number<1>{}); // 4
    static constexpr index_t WarpPerBlock_K1 = WarpPerBlock_1::at(number<2>{}); // 1
    static constexpr index_t Warp_M1         = WarpTile_1::at(number<0>{}); // 16
    static constexpr index_t Warp_N1         = WarpTile_1::at(number<1>{}); // 16
    static constexpr index_t Warp_K1         = WarpTile_1::at(number<2>{}); // 32

    static constexpr index_t ThreadPerBlock_M1 = Warp_M1 * WarpPerBlock_M1; // 16x1=16
    static constexpr index_t ThreadPerBlock_N1 = Warp_N1 * WarpPerBlock_N1; // 16x4=64
    static constexpr index_t ThreadPerBlock_K1 = Warp_K1 * WarpPerBlock_K1; // 32x1=32
    static_assert(Block_M1 % ThreadPerBlock_M1 == 0);
    static_assert(Block_N1 % ThreadPerBlock_N1 == 0);
    static_assert(Block_K1 % ThreadPerBlock_K1 == 0);
    static constexpr index_t Repeat_M1 = Block_M1 / ThreadPerBlock_M1; // 32/16=2
    static constexpr index_t Repeat_N1 = Block_N1 / ThreadPerBlock_N1; // 128/64=2
    static constexpr index_t Repeat_K1 = Block_K1 / ThreadPerBlock_K1; // 512/32=16

    static constexpr index_t BlockSize = get_warp_size() * NumWarps; // 32x4=128

    // some assert
    static_assert(Block_M0 == Block_M1);
    static_assert(Block_N0 == Block_K1 || (Block_N0 / 2) == Block_K1); // Gate Only or Gate+Up

    // pre-shuffle tile size compute (assume only for B matrix)
    // we flatten the each wave tile to a 1d linear tensor(at model loading time)
    // e.g. originally we have Block_N*Block_K tile size, after pre-shuffle
    // we can have Block_Nr*Block_Kr*Block_W, where Block_W is Warp_N*Warp_K,
    // and Block_Nr=Block_N/Warp_N, Block_Kr=Block_K/Warp_K
    static constexpr index_t Block_W0  = Warp_N0 * Warp_K0; // 16x32=512
    static constexpr index_t Block_Nr0 = Block_N0 / Warp_N0; // 512/16=32
    static constexpr index_t Block_Kr0 = Block_K0 / Warp_K0; // 128/32=4
    static constexpr index_t Block_W1  = Warp_N1 * Warp_K1; // 16x32=512
    static constexpr index_t Block_Nr1 = Block_N1 / Warp_N1; // 128/16=8
    static constexpr index_t Block_Kr1 = Block_K1 / Warp_K1; // 512/32=16

    static_assert(Block_W0 == Block_W1);
    // static_assert(Block_Nr0 == Block_Kr1);
};
} // namespace ck_tile
