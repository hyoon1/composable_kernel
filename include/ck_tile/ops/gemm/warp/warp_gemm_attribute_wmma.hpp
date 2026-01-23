// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl.hpp"

namespace ck_tile {

// TODO: currently only support 16 bit input, which means only support tr16_b128; will use ADataType
// to determine the layout in the future
template <typename Impl>
struct AWarpDstrEncodingTrait
{
    using type = tile_distribution_encoding<
        sequence<Impl::kRepeat>,
        tuple<sequence<Impl::kAMLane>,
              sequence<Impl::kABKLane, Impl::kABKPerLane>>,
        tuple<typename Impl::kABPs2RHssMajor>,
        tuple<typename Impl::kABPs2RHssMinor>,
        typename Impl::kABYs2RHsMajor,
        typename Impl::kABYs2RHsMinor>;
};

template <typename Impl>
struct BWarpDstrEncodingTrait
{
    using type = tile_distribution_encoding<
        sequence<Impl::kRepeat>,
        tuple<sequence<Impl::kBNLane>,
              sequence<Impl::kABKLane, Impl::kABKPerLane>>,
        tuple<typename Impl::kABPs2RHssMajor>,
        tuple<typename Impl::kABPs2RHssMinor>,
        typename Impl::kABYs2RHsMajor,
        typename Impl::kABYs2RHsMinor>;
};

template <typename Impl>
struct CWarpDstrEncodingTrait
{
    using type = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>,
              sequence<Impl::kCNLane>>,
        tuple<typename Impl::kCPs2RHssMajor>,
        tuple<typename Impl::kCPs2RHssMinor>,
        typename Impl::kCYs2RHsMajor,
        typename Impl::kCYs2RHsMinor>;
};

template <typename WarpGemmAttributeWmmaImpl_, bool kTransC = false>
struct WarpGemmAttributeWmma
{
    using Impl = remove_cvref_t<WarpGemmAttributeWmmaImpl_>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename Impl::AVecType;
    using BVecType = typename Impl::BVecType;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM          = Impl::kM;
    static constexpr index_t kN          = Impl::kN;
    static constexpr index_t kK          = Impl::kK;
    //static constexpr index_t kKPerThread = Impl::kABK0PerLane * Impl::kABK1PerLane;
    static constexpr index_t kKPerThread = Impl::kABKPerLane;

    CK_TILE_HOST_DEVICE static constexpr auto get_num_of_access() { return 1; }

    // 16 bit input, kAMLane = 16, kABK0PerLane = 4, kABKLane = 2, kABK1PerLane = 2
    // 8  bit input, kAMLane = 16, kABK0PerLane = 2, kABKLane = 2, kABK1PerLane = 4
    using AWarpDstrEncoding = typename AWarpDstrEncodingTrait<Impl>::type;
    using BWarpDstrEncoding = typename BWarpDstrEncodingTrait<Impl>::type;

    // kCM0PerLane = 4, kCMLane = 2, kCM1PerLane = 2, kCNLane = 16 for 16 bit input
    // kCM0PerLane = 2, kCMLane = 2, kCM1PerLane = 4, kCNLane = 16 for 8 bit input
    using CWarpDstrEncoding = typename CWarpDstrEncodingTrait<Impl>::type;

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        if constexpr(kTransC)
        {
            Impl{}(c_vec, b_vec, a_vec, bool_constant<post_nop_>{});
        }
        else
        {
            Impl{}(c_vec, a_vec, b_vec, bool_constant<post_nop_>{});
        }
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        if constexpr(kTransC)
        {
            return Impl{}(b_vec, a_vec);
        }
        else
        {
            return Impl{}(a_vec, b_vec);
        }
    }
};

template <typename WarpGemmAttributeWmmaImpl_, index_t kKIter,
          bool kTransC = false,
          WGAttrNumAccessEnum AttrNumAccess_ = WGAttrNumAccessEnum::Single>
struct WarpGemmAttributeWmmaIterateK
{
    static_assert(kKIter > 0, "wrong!");

    using Impl                           = remove_cvref_t<WarpGemmAttributeWmmaImpl_>;
    static constexpr auto AttrNumAccess  = AttrNumAccess_;
    static constexpr auto AttrNumAccessV = static_cast<index_t>(AttrNumAccess);

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AVecType =
        ext_vector_t<ADataType, vector_traits<typename Impl::AVecType>::vector_size * kKIter>;
    using BVecType =
        ext_vector_t<BDataType, vector_traits<typename Impl::BVecType>::vector_size * kKIter>;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM          = Impl::kM;
    static constexpr index_t kN          = Impl::kN;
    static constexpr index_t kK          = Impl::kK * kKIter;
    static constexpr index_t kKPerThread = Impl::kABKPerLane * kKIter;

    CK_TILE_HOST_DEVICE static constexpr auto get_num_of_access() { return kKIter; }

    static_assert(Impl::kAMBlock == 1 || Impl::kBNBlock == 1,
                  "Multi-block on both M & N directions is not supported");

    CK_TILE_DEVICE static constexpr auto get_awarp_dstr_encoding()
    {
        //printf("[WarpGemmAttributeWmmaIterateK Debug] 0\n");
        if constexpr(Impl::kAMBlock == 1 && Impl::kBNBlock == 1)
        {
            if constexpr(AttrNumAccessV == 1)
            {
                printf("[WarpGemmAttributeWmmaIterateK Debug] 1\n");
                return tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<Impl::kAMLane>,
                          sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
                    tuple<sequence<2, 1>>,
                    tuple<sequence<0, 0>>,
                    sequence<2>,
                    sequence<1>>{};
            }
            else
            {
                printf("[WarpGemmAttributeWmmaIterateK Debug] 2\n");
                static_assert(kKPerThread % AttrNumAccessV == 0,
                              "kKPerThread must be divisible by NumAccess");
                return tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<Impl::kAMLane>,
                          sequence<AttrNumAccessV,
                                   Impl::kABKLane,
                                   Impl::kABKPerLane * kKIter / AttrNumAccessV>>,
                    tuple<sequence<2, 1>>,
                    tuple<sequence<1, 0>>,
                    sequence<2, 2>,
                    sequence<0, 2>>{};
            }
        }
        else if constexpr(Impl::kAMBlock == 1 && 1 < Impl::kBNBlock)
        {
            printf("[WarpGemmAttributeWmmaIterateK Debug] 3\n");
            static_assert(AttrNumAccessV == 1,
                          "Multiple access is not supported when using multi-block");
            // each M blocks share the same data
            return tile_distribution_encoding<
                sequence<Impl::kBNBlock>,
                tuple<sequence<Impl::kAMLane>,
                      sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
                tuple<sequence<0, 2, 1>>,
                tuple<sequence<0, 0, 0>>,
                sequence<2>,
                sequence<1>>{};
        }
        else if constexpr(1 < Impl::kAMBlock && Impl::kBNBlock == 1)
        {
            printf("[WarpGemmAttributeWmmaIterateK Debug] 4\n");
            static_assert(AttrNumAccessV == 1,
                          "Multiple access is not supported when using multi-block");
            // single block to multi-block thread mapping
            return tile_distribution_encoding<
                sequence<>,
                tuple<sequence<Impl::kAMBlock, Impl::kAMLane>,
                      sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
                tuple<sequence<1, 2, 1>>,
                tuple<sequence<0, 0, 1>>,
                sequence<2>,
                sequence<1>>{};
        }
        else
        {
            printf("[WarpGemmAttributeWmmaIterateK Debug] 5\n");
        }
    }

    CK_TILE_DEVICE static constexpr auto get_bwarp_dstr_encoding()
    {
        if constexpr(Impl::kAMBlock == 1 && Impl::kBNBlock == 1)
        {
            if constexpr(AttrNumAccessV == 1)
            {
                return tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<Impl::kBNLane>,
                          sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
                    tuple<sequence<2, 1>>,
                    tuple<sequence<0, 0>>,
                    sequence<2>,
                    sequence<1>>{};
            }
            else
            {

                static_assert(kKPerThread % AttrNumAccessV == 0,
                              "kKPerThread must be divisible by NumAccess");
                return tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<Impl::kBNLane>,
                          sequence<AttrNumAccessV,
                                   Impl::kABKLane,
                                   Impl::kABKPerLane * kKIter / AttrNumAccessV>>,
                    tuple<sequence<2, 1>>,
                    tuple<sequence<1, 0>>,
                    sequence<2, 2>,
                    sequence<0, 2>>{};
            }
        }
        else if constexpr(Impl::kAMBlock == 1 && 1 < Impl::kBNBlock)
        {
            static_assert(AttrNumAccessV == 1,
                          "Multiple access is not supported when using multi-block");
            // single block to multi-block thread mapping
            return tile_distribution_encoding<
                sequence<>,
                tuple<sequence<Impl::kBNBlock, Impl::kBNLane>,
                      sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
                tuple<sequence<1, 2, 1>>,
                tuple<sequence<0, 0, 1>>,
                sequence<2>,
         
         sequence<1>>{};
        }
        else if constexpr(1 < Impl::kAMBlock && Impl::kBNBlock == 1)
        {
            static_assert(AttrNumAccessV == 1,
                          "Multiple access is not supported when using multi-block");
            // each N blocks share the same data
            return tile_distribution_encoding<
                sequence<Impl::kAMBlock>,
                tuple<sequence<Impl::kBNLane>,
                      sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
                tuple<sequence<0, 2, 1>>,
                tuple<sequence<0, 0, 0>>,
                sequence<2>,
                sequence<1>>{};
        }
    }

    CK_TILE_DEVICE static constexpr auto get_cwarp_dstr_encoding()
    {
        if constexpr(Impl::kAMBlock == 1 && Impl::kBNBlock == 1)
        {
            return tile_distribution_encoding<
                sequence<>,
                tuple<sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>,
                      sequence<Impl::kCNLane>>,
                tuple<sequence<1, 2>>,
                tuple<sequence<1, 0>>,
                sequence<1, 1>,
                sequence<0, 2>>{};
        }
        else if constexpr(Impl::kAMBlock == 1 && 1 < Impl::kBNBlock)
        {
            return tile_distribution_encoding<
                sequence<>,
                tuple<sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>,
                      sequence<Impl::kBNBlock * Impl::kCNLane>>,
                tuple<sequence<1, 2>>,
                tuple<sequence<1, 0>>,
                sequence<1, 1>,
                sequence<0, 2>>{};
        }
        else if constexpr(1 < Impl::kAMBlock && Impl::kBNBlock == 1)
        {
            return tile_distribution_encoding<
                sequence<>,
                tuple<
                    sequence<Impl::kCM0PerLane, Impl::kAMBlock * Impl::kCMLane, Impl::kCM1PerLane>,
                    sequence<Impl::kCNLane>>,
                tuple<sequence<1, 2>>,
                tuple<sequence<1, 0>>,
                sequence<1, 1>,
                sequence<0, 2>>{};
        }
    }

    using AWarpDstrEncoding = decltype(get_awarp_dstr_encoding());

    using BWarpDstrEncoding = decltype(get_bwarp_dstr_encoding());

    using CWarpDstrEncoding = decltype(get_cwarp_dstr_encoding());

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        //printf("[WarpGemmAttributeWmmaIterateK Debug] 6\n");
        using buf_a = thread_buffer<typename Impl::AVecType, kKIter>;
        using buf_b = thread_buffer<typename Impl::BVecType, kKIter>;

        static_for<0, kKIter, 1>{}([&](auto iKIter) {
            Impl{}(c_vec,
                   reinterpret_cast<const buf_a&>(a_vec)
                       .template get_as<typename Impl::AVecType>()[iKIter],
                   reinterpret_cast<const buf_b&>(b_vec)
                       .template get_as<typename Impl::BVecType>()[iKIter],
                   bool_constant<post_nop_>{});
        });
    }

    template <index_t iKIter, bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   number<iKIter>,
                                   bool_constant<post_nop_> = {}) const
    {
        printf("[WarpGemmAttributeWmmaIterateK Debug] 7\n");
        using buf_a = thread_buffer<typename Impl::AVecType, kKIter>;
        using buf_b = thread_buffer<typename Impl::BVecType, kKIter>;

        static_assert(iKIter < kKIter);

        // static_for<0, kKIter, 1>{}([&](auto iKIter) {
        Impl{}(c_vec,
               reinterpret_cast<const buf_a&>(a_vec)
                   .template get_as<typename Impl::AVecType>()[iKIter],
               reinterpret_cast<const buf_b&>(b_vec)
                   .template get_as<typename Impl::BVecType>()[iKIter],
               bool_constant<post_nop_>{});
        //});
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        printf("[WarpGemmAttributeWmmaIterateK Debug] 8\n");
        constexpr auto I0 = number<0>{};
        using buf_a       = thread_buffer<typename Impl::AVecType, kKIter>;
        using buf_b       = thread_buffer<typename Impl::BVecType, kKIter>;

        // c = a * b
        auto c_vec = Impl{}(
            reinterpret_cast<const buf_a&>(a_vec).template get_as<typename Impl::AVecType>()[I0],
            reinterpret_cast<const buf_b&>(b_vec).template get_as<typename Impl::BVecType>()[I0]);

        // c += a * b
        static_for<1, kKIter, 1>{}([&](auto iKIter) {
            Impl{}(c_vec,
                   reinterpret_cast<const buf_a&>(a_vec)
                       .template get_as<typename Impl::AVecType>()[iKIter],
                   reinterpret_cast<const buf_b&>(b_vec)
                       .template get_as<typename Impl::BVecType>()[iKIter]);
        });

        return c_vec;
    }
};

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          index_t M_Warp_Tile,
          index_t N_Warp_Tile,
          index_t K_Warp_Tile>
CK_TILE_HOST bool check_wmma_supported()
{
    if(is_gfx12_supported())
    {
        return has_wmma_traits_v<gfx12_t,
                                 ADataType,
                                 BDataType,
                                 AccDataType,
                                 M_Warp_Tile,
                                 N_Warp_Tile,
                                 K_Warp_Tile>;
    }
    else if(is_gfx11_supported())
    {
        return has_wmma_traits_v<gfx11_t,
                                 ADataType,
                                 BDataType,
                                 AccDataType,
                                 M_Warp_Tile,
                                 N_Warp_Tile,
                                 K_Warp_Tile>;
    }
    else
    {
        return false;
    }
}

} // namespace ck_tile
