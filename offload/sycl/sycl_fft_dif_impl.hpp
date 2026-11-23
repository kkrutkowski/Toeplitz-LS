// Multi-includable Triple-Precision SYCL DIF (Sande-Tukey) Radix-2 FFT Kernels & Plan.
// Intentionally has no #pragma once; included 3x by sycl_fft_dif.hpp with
// sycl_fft_precision.hpp configured for FP32 (f), FP64 (raw), and DF (df).

namespace sycl_fft {

// ============================================================================
// Pure Radix-2 DIF stages in registers on E = 2^LogE elements (no inter-thread
// twiddles; exact compile-time W_32 constants with W_2, W_4, W_8 specializations).
// ============================================================================
template <int LogE, int Sign>
SYCL_FFT_FORCE_INLINE void SYCL_FFT_SUFFIX(dif_radix2_reg_base)(
    SYCL_FFT_REAL (&r_re)[1 << LogE], SYCL_FFT_REAL (&r_im)[1 << LogE]) {
    using Real = SYCL_FFT_REAL;
    constexpr int E = 1 << LogE;
    static_assert(LogE <= 5, "W_32 table supports up to LogE == 5 (E == 32)");

    unroll_for<LogE>([&](auto s_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int S = s_ic.value;
        constexpr int H = E >> (S + 1);
        constexpr int Step = H << 1;
        constexpr int Groups = E / Step;

        unroll_for<H>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int M = m_ic.value;
            constexpr int Idx32 = M * (32 / Step);

            unroll_for<Groups>([&](auto g_ic) SYCL_FFT_LAMBDA_INLINE {
                constexpr int A = g_ic.value * Step + M;
                constexpr int B = A + H;
                const Real ar = r_re[A], ai = r_im[A];
                const Real br = r_re[B], bi = r_im[B];
                const Real dr = SYCL_FFT_SUB(ar, br);
                const Real di = SYCL_FFT_SUB(ai, bi);
                r_re[A] = SYCL_FFT_ADD(ar, br);
                r_im[A] = SYCL_FFT_ADD(ai, bi);

                if constexpr (M == 0) {
                    r_re[B] = dr;
                    r_im[B] = di;
                } else if constexpr (Idx32 == 8) {
                    if constexpr (Sign < 0) {
                        r_re[B] = di;
                        r_im[B] = SYCL_FFT_NEG(dr);
                    } else {
                        r_re[B] = SYCL_FFT_NEG(di);
                        r_im[B] = dr;
                    }
                } else if constexpr (Idx32 == 4) {
                    SYCL_FFT_CMUL_SQRT1_2(false, Sign < 0, dr, di, r_re[B], r_im[B]);
                } else if constexpr (Idx32 == 12) {
                    SYCL_FFT_CMUL_SQRT1_2(true, Sign < 0, dr, di, r_re[B], r_im[B]);
                } else {
                    constexpr Real cr = SYCL_FFT_TWIDDLE32_RE(Sign, Idx32);
                    constexpr Real ci = SYCL_FFT_TWIDDLE32_IM(Sign, Idx32);
                    SYCL_FFT_CMUL(dr, di, cr, ci, r_re[B], r_im[B]);
                }
            });
        });
    });
}

// ============================================================================
// Factored Bit-Reversed Twiddle Application:
//   Multiplies slot P (0..E-1) by W_{E*T}^{t * bitrev_{LogE}(P)} in strictly
//   ascending P order and invokes on_elem(p_ic, out_re, out_im).
//   Eliminates 12.5-37% of complex multiplies vs unfactored DIF stages and
//   evaluates only ONE octant-1 polynomial trig call (t / (E*T) < 1/8 for E >= 8).
// ============================================================================
template <int LogE, int LogT, int Sign, typename OnElem>
SYCL_FFT_FORCE_INLINE void SYCL_FFT_SUFFIX(apply_bitrev_twiddles_with)(
    const SYCL_FFT_REAL (&r_re)[1 << LogE],
    const SYCL_FFT_REAL (&r_im)[1 << LogE],
    int t,
    OnElem&& on_elem) {
    using Real = SYCL_FFT_REAL;
    using AngleT = SYCL_FFT_ANGLE_T;
    constexpr int E = 1 << LogE;
    constexpr int T = 1 << LogT;
    constexpr AngleT inv_m = static_cast<AngleT>(1.0) / static_cast<AngleT>(E * T);

    on_elem(std::integral_constant<int, 0>{}, r_re[0], r_im[0]);

    Real w1r, w1i;
    if constexpr (LogE >= 4 && LogE + LogT <= 8) {
        // Exact small-angle table lookup (W_64, W_128, W_256) -> zero trig ops!
        constexpr int Scale = 256 / (E * T);
        const int idx = t * Scale;
        w1r = SYCL_FFT_TWIDDLE256_RE(Sign, idx);
        w1i = SYCL_FFT_TWIDDLE256_IM(Sign, idx);
    } else if constexpr (LogE >= 3) {
        SYCL_FFT_SINCOS2PI_OCT1(Sign, static_cast<AngleT>(t) * inv_m, w1r, w1i);
    } else {
        SYCL_FFT_SINCOS2PI_HALF(Sign, static_cast<AngleT>(t) * inv_m, w1r, w1i);
    }
    Real w2r, w2i, w3r, w3i, w4r, w4i;
    SYCL_FFT_CSQR(w1r, w1i, w2r, w2i);
    SYCL_FFT_CMUL(w2r, w2i, w1r, w1i, w3r, w3i);
    SYCL_FFT_CSQR(w2r, w2i, w4r, w4i);

    if constexpr (LogE == 3) {
        const Real lo_r[4] = {SYCL_FFT_ONE, w2r, w1r, w3r};
        const Real lo_i[4] = {SYCL_FFT_ZERO, w2i, w1i, w3i};
        unroll_for<7>([&](auto idx_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int P = idx_ic.value + 1;
            constexpr int A = P >> 1; // 0..3
            constexpr int B = P & 1;  // 0..1 -> {1, w4}
            Real wr, wi;
            if constexpr (A == 0) {
                wr = w4r;
                wi = w4i;
            } else if constexpr (B == 0) {
                wr = lo_r[A];
                wi = lo_i[A];
            } else {
                SYCL_FFT_CMUL(lo_r[A], lo_i[A], w4r, w4i, wr, wi);
            }
            Real or_, oi;
            SYCL_FFT_CMUL(r_re[P], r_im[P], wr, wi, or_, oi);
            on_elem(std::integral_constant<int, P>{}, or_, oi);
        });
    } else if constexpr (LogE == 4) {
        Real w8r, w8i, w12r, w12i;
        SYCL_FFT_CSQR(w4r, w4i, w8r, w8i);
        SYCL_FFT_CMUL(w8r, w8i, w4r, w4i, w12r, w12i);
        const Real lo_r[4] = {SYCL_FFT_ONE, w2r, w1r, w3r};
        const Real lo_i[4] = {SYCL_FFT_ZERO, w2i, w1i, w3i};
        const Real hi_r[4] = {SYCL_FFT_ONE, w8r, w4r, w12r};
        const Real hi_i[4] = {SYCL_FFT_ZERO, w8i, w4i, w12i};
        unroll_for<15>([&](auto idx_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int P = idx_ic.value + 1;
            constexpr int A = P >> 2;
            constexpr int B = P & 3;
            Real wr, wi;
            if constexpr (A == 0) {
                wr = hi_r[B];
                wi = hi_i[B];
            } else if constexpr (B == 0) {
                wr = lo_r[A];
                wi = lo_i[A];
            } else {
                SYCL_FFT_CMUL(lo_r[A], lo_i[A], hi_r[B], hi_i[B], wr, wi);
            }
            Real or_, oi;
            SYCL_FFT_CMUL(r_re[P], r_im[P], wr, wi, or_, oi);
            on_elem(std::integral_constant<int, P>{}, or_, oi);
        });
    } else if constexpr (LogE == 5) {
        Real w8r, w8i, w16r, w16i, w24r, w24i;
        SYCL_FFT_CSQR(w4r, w4i, w8r, w8i);
        SYCL_FFT_CSQR(w8r, w8i, w16r, w16i);
        SYCL_FFT_CMUL(w16r, w16i, w8r, w8i, w24r, w24i);
        const Real hi_r[4] = {SYCL_FFT_ONE, w16r, w8r, w24r};
        const Real hi_i[4] = {SYCL_FFT_ZERO, w16i, w8i, w24i};

#if SYCL_FFT_WIDE
        // Minimal-register factorization for 8-byte reals (FP64/DF):
        // compute lo_a on the fly per A = 0..7 using only {w1, w2, w3, w4} + hi[4].
        const Real base_r[4] = {SYCL_FFT_ONE, w2r, w1r, w3r};
        const Real base_i[4] = {SYCL_FFT_ZERO, w2i, w1i, w3i};
        unroll_for<8>([&](auto a_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int A = a_ic.value;
            constexpr int Ahi = A >> 1;
            constexpr int A0 = A & 1;
            Real lor, loi;
            if constexpr (A == 0) {
                lor = SYCL_FFT_ONE;
                loi = SYCL_FFT_ZERO;
            } else if constexpr (Ahi == 0) {
                lor = w4r;
                loi = w4i;
            } else if constexpr (A0 == 0) {
                lor = base_r[Ahi];
                loi = base_i[Ahi];
            } else {
                SYCL_FFT_CMUL(base_r[Ahi], base_i[Ahi], w4r, w4i, lor, loi);
            }
            unroll_for<4>([&](auto b_ic) SYCL_FFT_LAMBDA_INLINE {
                constexpr int B = b_ic.value;
                constexpr int P = (A << 2) | B;
                if constexpr (P > 0) {
                    Real wr, wi;
                    if constexpr (A == 0) {
                        wr = hi_r[B];
                        wi = hi_i[B];
                    } else if constexpr (B == 0) {
                        wr = lor;
                        wi = loi;
                    } else {
                        SYCL_FFT_CMUL(lor, loi, hi_r[B], hi_i[B], wr, wi);
                    }
                    Real or_, oi;
                    SYCL_FFT_CMUL(r_re[P], r_im[P], wr, wi, or_, oi);
                    on_elem(std::integral_constant<int, P>{}, or_, oi);
                }
            });
        });
#else
        Real w5r, w5i, w6r, w6i, w7r, w7i;
        SYCL_FFT_CMUL(w4r, w4i, w1r, w1i, w5r, w5i);
        SYCL_FFT_CMUL(w4r, w4i, w2r, w2i, w6r, w6i);
        SYCL_FFT_CMUL(w4r, w4i, w3r, w3i, w7r, w7i);
        const Real lo_r[8] = {SYCL_FFT_ONE, w4r, w2r, w6r, w1r, w5r, w3r, w7r};
        const Real lo_i[8] = {SYCL_FFT_ZERO, w4i, w2i, w6i, w1i, w5i, w3i, w7i};
        unroll_for<31>([&](auto idx_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int P = idx_ic.value + 1;
            constexpr int A = P >> 2;
            constexpr int B = P & 3;
            Real wr, wi;
            if constexpr (A == 0) {
                wr = hi_r[B];
                wi = hi_i[B];
            } else if constexpr (B == 0) {
                wr = lo_r[A];
                wi = lo_i[A];
            } else {
                SYCL_FFT_CMUL(lo_r[A], lo_i[A], hi_r[B], hi_i[B], wr, wi);
            }
            Real or_, oi;
            SYCL_FFT_CMUL(r_re[P], r_im[P], wr, wi, or_, oi);
            on_elem(std::integral_constant<int, P>{}, or_, oi);
        });
#endif
    }
}

template <int LogE, int LogT, int Sign>
SYCL_FFT_FORCE_INLINE void SYCL_FFT_SUFFIX(dif_radix2_reg)(
    SYCL_FFT_REAL (&r_re)[1 << LogE], SYCL_FFT_REAL (&r_im)[1 << LogE], int t) {
    SYCL_FFT_SUFFIX(dif_radix2_reg_base)<LogE, Sign>(r_re, r_im);
    if constexpr (LogT > 0) {
        SYCL_FFT_SUFFIX(apply_bitrev_twiddles_with)<LogE, LogT, Sign>(
            r_re, r_im, t, [&](auto p_ic, SYCL_FFT_REAL vr, SYCL_FFT_REAL vi) SYCL_FFT_LAMBDA_INLINE {
                constexpr int P = p_ic.value;
                if constexpr (P > 0) {
                    r_re[P] = vr;
                    r_im[P] = vi;
                }
            });
    }
}

// ============================================================================
// Warp-Convergent Sub-Group Radix-2 DIF across W = 2^LogW lanes (LogW <= 5):
//   Unified branchless butterfly (single ADD across all 32 lanes via sign-bit XOR),
//   single polynomial trig call for widest stage + negated squaring for H = 8, 4.
// ============================================================================
template <int LogW, int E, int Sign>
SYCL_FFT_FORCE_INLINE void SYCL_FFT_SUFFIX(dif_radix2_subgroup)(
    const sycl::sub_group& sg, SYCL_FFT_REAL (&r_re)[E], SYCL_FFT_REAL (&r_im)[E], int lane) {
    using Real = SYCL_FFT_REAL;
    using AngleT = SYCL_FFT_ANGLE_T;
    constexpr int W = 1 << LogW;
    const int l = lane & (W - 1);

    unroll_for<LogW>([&](auto s_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int S = s_ic.value;
        constexpr int H = W >> (S + 1);
        const bool is_hi = (l & H) != 0;
        const int k = l & (H - 1);

        Real tw_re = SYCL_FFT_ONE;
        Real tw_im = SYCL_FFT_ZERO;
        if constexpr (H > 2) {
            // Exact W_32 table lookup (W <= 32 -> zero polynomial trig & zero squaring!)
            const int idx32 = k * (16 / H);
            tw_re = SYCL_FFT_TWIDDLE32_RE(Sign, idx32);
            tw_im = SYCL_FFT_TWIDDLE32_IM(Sign, idx32);
        }
        unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
            constexpr int M = m_ic.value;
            const Real my_re = r_re[M];
            const Real my_im = r_im[M];
            const Real p_re = SYCL_FFT_SHUFFLE_XOR(sg, my_re, H);
            const Real p_im = SYCL_FFT_SHUFFLE_XOR(sg, my_im, H);

            // Unified warp-convergent add/sub: p + (is_hi ? -my : my)
            const Real s_re = SYCL_FFT_COND_NEG(my_re, is_hi);
            const Real s_im = SYCL_FFT_COND_NEG(my_im, is_hi);
            Real u_re = SYCL_FFT_ADD(p_re, s_re);
            Real u_im = SYCL_FFT_ADD(p_im, s_im);

            if constexpr (H == 1) {
                r_re[M] = u_re;
                r_im[M] = u_im;
            } else if constexpr (H == 2) {
                if (is_hi && k != 0) {
                    if constexpr (Sign < 0) {
                        r_re[M] = u_im;
                        r_im[M] = SYCL_FFT_NEG(u_re);
                    } else {
                        r_re[M] = SYCL_FFT_NEG(u_im);
                        r_im[M] = u_re;
                    }
                } else {
                    r_re[M] = u_re;
                    r_im[M] = u_im;
                }
            } else {
                if (is_hi) {
                    Real tr, ti;
                    SYCL_FFT_CMUL(u_re, u_im, tw_re, tw_im, tr, ti);
                    r_re[M] = tr;
                    r_im[M] = ti;
                } else {
                    r_re[M] = u_re;
                    r_im[M] = u_im;
                }
            }
        });
    });
}

// ============================================================================
// Single-Pass Kernel A (Split smem_re / smem_im, 1 barrier, <= 34 KiB smem):
//   FP32:     logN = 10 (E=8, B=2), 11 (E=8, B=1), 12 (E=16, B=1)
//   FP64/DF:  logN = 10 (E=16, B=1, 16 KiB XOR-swizzled bank-conflict-free)
// ============================================================================
template <int LogN, int LogE, int B, int Sign>
void SYCL_FFT_SUFFIX(launch_single_pass_split_smem)(
    sycl::queue& q, const SYCL_FFT_COMPLEX* in, SYCL_FFT_COMPLEX* out, size_t bs) {
    using Real = SYCL_FFT_REAL;
    using ComplexT = SYCL_FFT_COMPLEX;
    using SmemT = SYCL_FFT_SMEM_T;

    constexpr int N = 1 << LogN;
    constexpr int E = 1 << LogE;
    constexpr int LogT = LogN - LogE;
    constexpr int T = 1 << LogT;
    constexpr int LogW = LogT - LogE;
    static_assert(LogW >= 0 && LogW <= 5, "T_sub must fit within a 32-lane sub-group");
    constexpr int W = 1 << LogW;
    // Wide (8B smem) uses zero-padding XOR swizzle to eliminate 4-way bank conflicts
    // while keeping total smem at exactly 16 KiB (preserving full workgroup occupancy).
    constexpr bool UseXorSwizzle = (SYCL_FFT_WIDE != 0) && (W < 16);
    constexpr int Pad = (SYCL_FFT_WIDE || W == 32) ? 0 : 16;
    constexpr int RowStride = T + Pad;
    constexpr int FftSmem = E * RowStride;
    constexpr int WG = B * T;

    if (bs == 0) return;
    const size_t num_wg = (bs + B - 1) / B;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<SmemT, 1> smem_re(sycl::range<1>(B * FftSmem), h);
        sycl::local_accessor<SmemT, 1> smem_im(sycl::range<1>(B * FftSmem), h);

        h.parallel_for(
            sycl::nd_range<1>(num_wg * WG, WG),
            [=](sycl::nd_item<1> item) SYCL_FFT_LAMBDA_INLINE {
                const size_t wg_id = item.get_group_linear_id();
                const int tid = static_cast<int>(item.get_local_linear_id());
                const int b_local = tid >> LogT;
                const int t = tid & (T - 1);
                const size_t batch = wg_id * B + static_cast<size_t>(b_local);
                const bool valid = (B == 1) || (batch < bs);

                const ComplexT* g_in = in + batch * N;
                ComplexT* g_out = out + batch * N;

                Real r_re[E], r_im[E];
                if (valid) {
                    unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                        constexpr int M = m_ic.value;
                        SYCL_FFT_LOAD_COMPLEX(g_in, t + M * T, r_re[M], r_im[M]);
                    });
                } else {
                    unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                        constexpr int M = m_ic.value;
                        r_re[M] = SYCL_FFT_ZERO;
                        r_im[M] = SYCL_FFT_ZERO;
                    });
                }

                SYCL_FFT_SUFFIX(dif_radix2_reg)<LogE, LogT, Sign>(r_re, r_im, t);

                const int smem_base = b_local * FftSmem;
                const int sub = t >> LogW;
                const int l = t & (W - 1);
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    const int w_idx = UseXorSwizzle
                        ? (smem_base + M * T + ((sub ^ M) << LogW) + l)
                        : (smem_base + M * RowStride + t);
                    smem_re[w_idx] = SYCL_FFT_TO_SMEM(r_re[M]);
                    smem_im[w_idx] = SYCL_FFT_TO_SMEM(r_im[M]);
                });

                sycl::group_barrier(item.get_group());

                const int row_off = smem_base + sub * RowStride + l;
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    const int r_idx = UseXorSwizzle
                        ? (smem_base + sub * T + ((M ^ sub) << LogW) + l)
                        : (row_off + M * W);
                    r_re[M] = SYCL_FFT_FROM_SMEM(smem_re[r_idx]);
                    r_im[M] = SYCL_FFT_FROM_SMEM(smem_im[r_idx]);
                });

                SYCL_FFT_SUFFIX(dif_radix2_reg)<LogE, LogW, Sign>(r_re, r_im, l);

                auto sg = item.get_sub_group();
                SYCL_FFT_SUFFIX(dif_radix2_subgroup)<LogW, E, Sign>(sg, r_re, r_im, l);

                if (valid) {
                    const int out_base = (sub << LogE) * W + l;
                    unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                        constexpr int M = m_ic.value;
                        SYCL_FFT_STORE_COMPLEX(g_out, out_base + M * W, r_re[M], r_im[M]);
                    });
                }
            });
    });
}

// ============================================================================
// Single-Pass Kernel B (Sequential real-then-imag 32 KiB smem):
//   FP32:    logN = 13 (T=512, E=16, W=32, 32 KiB)
//   FP64/DF: logN = 12 (T=256, E=16, W=16, 32 KiB)
// ============================================================================
template <int LogN, int LogE, int Sign>
void SYCL_FFT_SUFFIX(launch_single_pass_seq_smem)(
    sycl::queue& q, const SYCL_FFT_COMPLEX* in, SYCL_FFT_COMPLEX* out, size_t bs) {
    using Real = SYCL_FFT_REAL;
    using ComplexT = SYCL_FFT_COMPLEX;
    using SmemT = SYCL_FFT_SMEM_T;

    constexpr int N = 1 << LogN;
    constexpr int E = 1 << LogE;
    constexpr int LogT = LogN - LogE;
    constexpr int T = 1 << LogT;
    constexpr int LogW = LogT - LogE;
    static_assert(LogW >= 0 && LogW <= 5, "W must fit within a 32-lane sub-group");
    constexpr int W = 1 << LogW;
    constexpr int SmemElems = E * T; // 32 KiB

    if (bs == 0) return;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<SmemT, 1> smem(sycl::range<1>(SmemElems), h);

        h.parallel_for(
            sycl::nd_range<1>(bs * T, T),
            [=](sycl::nd_item<1> item) SYCL_FFT_LAMBDA_INLINE {
                const size_t batch = item.get_group_linear_id();
                const int t = static_cast<int>(item.get_local_linear_id());
                const ComplexT* g_in = in + batch * N;
                ComplexT* g_out = out + batch * N;

                Real r_re[E], r_im[E];
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    SYCL_FFT_LOAD_COMPLEX(g_in, t + M * T, r_re[M], r_im[M]);
                });

                SYCL_FFT_SUFFIX(dif_radix2_reg)<LogE, LogT, Sign>(r_re, r_im, t);

                const int sub = t >> LogW;
                const int l = t & (W - 1);

                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    smem[M * T + t] = SYCL_FFT_TO_SMEM(r_re[M]);
                });
                sycl::group_barrier(item.get_group());
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    r_re[M] = SYCL_FFT_FROM_SMEM(smem[sub * T + M * W + l]);
                });
                sycl::group_barrier(item.get_group());

                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    smem[M * T + t] = SYCL_FFT_TO_SMEM(r_im[M]);
                });
                sycl::group_barrier(item.get_group());
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    r_im[M] = SYCL_FFT_FROM_SMEM(smem[sub * T + M * W + l]);
                });

                SYCL_FFT_SUFFIX(dif_radix2_reg)<LogE, LogW, Sign>(r_re, r_im, l);

                auto sg = item.get_sub_group();
                SYCL_FFT_SUFFIX(dif_radix2_subgroup)<LogW, E, Sign>(sg, r_re, r_im, l);

                const int out_base = sub * T + l;
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    SYCL_FFT_STORE_COMPLEX(g_out, out_base + M * W, r_re[M], r_im[M]);
                });
            });
    });
}

// ============================================================================
// Single-Pass Chunked Kernel C (32 KiB smem):
//   FP32 (!SYCL_FFT_WIDE):    N = 16384 (T=512, E=32, 2 chunks of 16, seq smem)
//   FP64/DF (SYCL_FFT_WIDE):  N =  8192 (T=256, E=32, 4 chunks of 8, split smem)
// ============================================================================
#if !SYCL_FFT_WIDE

template <int C, int Sign>
SYCL_FFT_FORCE_INLINE void SYCL_FFT_SUFFIX(finish_chunk_16384)(
    const sycl::nd_item<1>& item,
    const sycl::sub_group& sg,
    const sycl::local_accessor<SYCL_FFT_SMEM_T, 1>& smem,
    const SYCL_FFT_REAL (&r_re)[32],
    const SYCL_FFT_REAL (&r_im)[32],
    SYCL_FFT_COMPLEX* g_out,
    int t,
    int sub,
    int l) {
    using Real = SYCL_FFT_REAL;
    constexpr int T = 512;
    constexpr int W = 32;
    constexpr int LogW = 5;
    constexpr int Chunk = 16;

    if constexpr (C > 0) {
        sycl::group_barrier(item.get_group());
    }
    Real c_re[Chunk], c_im[Chunk];
    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int M = m_ic.value;
        smem[M * T + t] = SYCL_FFT_TO_SMEM(r_re[C * Chunk + M]);
    });
    sycl::group_barrier(item.get_group());
    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int M = m_ic.value;
        c_re[M] = SYCL_FFT_FROM_SMEM(smem[sub * T + M * W + l]);
    });
    sycl::group_barrier(item.get_group());

    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int M = m_ic.value;
        smem[M * T + t] = SYCL_FFT_TO_SMEM(r_im[C * Chunk + M]);
    });
    sycl::group_barrier(item.get_group());
    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int M = m_ic.value;
        c_im[M] = SYCL_FFT_FROM_SMEM(smem[sub * T + M * W + l]);
    });

    SYCL_FFT_SUFFIX(dif_radix2_reg)<4, LogW, Sign>(c_re, c_im, l);
    SYCL_FFT_SUFFIX(dif_radix2_subgroup)<LogW, Chunk, Sign>(sg, c_re, c_im, l);

    const int out_base = (C * Chunk + sub) * T + l;
    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int M = m_ic.value;
        SYCL_FFT_STORE_COMPLEX(g_out, out_base + M * W, c_re[M], c_im[M]);
    });
}

template <int Sign>
void SYCL_FFT_SUFFIX(launch_single_pass_16384)(
    sycl::queue& q, const SYCL_FFT_COMPLEX* in, SYCL_FFT_COMPLEX* out, size_t bs) {
    using Real = SYCL_FFT_REAL;
    using ComplexT = SYCL_FFT_COMPLEX;
    using SmemT = SYCL_FFT_SMEM_T;

    constexpr int N = 16384;
    constexpr int T = 512;
    constexpr int E = 32;
    constexpr int LogE = 5;
    constexpr int LogT = 9;
    constexpr int LogW = 5;
    constexpr int W = 32;
    constexpr int Chunk = 16;
    constexpr int SmemElems = Chunk * T; // 32 KiB

    if (bs == 0) return;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<SmemT, 1> smem(sycl::range<1>(SmemElems), h);

        h.parallel_for(
            sycl::nd_range<1>(bs * T, T),
            [=](sycl::nd_item<1> item) SYCL_FFT_LAMBDA_INLINE {
                const size_t batch = item.get_group_linear_id();
                const int t = static_cast<int>(item.get_local_linear_id());
                const ComplexT* g_in = in + batch * N;
                ComplexT* g_out = out + batch * N;

                Real r_re[E], r_im[E];
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    SYCL_FFT_LOAD_COMPLEX(g_in, t + M * T, r_re[M], r_im[M]);
                });

                SYCL_FFT_SUFFIX(dif_radix2_reg)<LogE, LogT, Sign>(r_re, r_im, t);

                const int sub = t >> LogW;
                const int l = t & (W - 1);
                auto sg = item.get_sub_group();

                SYCL_FFT_SUFFIX(finish_chunk_16384)<0, Sign>(item, sg, smem, r_re, r_im, g_out, t, sub, l);
                SYCL_FFT_SUFFIX(finish_chunk_16384)<1, Sign>(item, sg, smem, r_re, r_im, g_out, t, sub, l);
            });
    });
}

#else // SYCL_FFT_WIDE: 100%-warp-efficient 2-transpose single-pass kernels for N = 2048, 4096, 8192

// Synchronize threads within a W-lane sub-tile (W = 8 or 16).
// Uses a sub-group barrier (bar.warp.sync on NVIDIA) whenever the device's sub-group
// covers the full tile (folds to an unconditional sub-group barrier on 32/64-lane GPUs),
// with a safe work-group fallback for narrower sub-groups.
template <int W>
SYCL_FFT_FORCE_INLINE void SYCL_FFT_SUFFIX(sync_sub_tile)(
    const sycl::nd_item<1>& item, const sycl::sub_group& sg) {
    if (sg.get_max_local_range()[0] >= static_cast<size_t>(W)) {
        sycl::group_barrier(sg);
    } else {
        sycl::group_barrier(item.get_group());
    }
}

// Single-Pass Wide N = 2048 (16 * 16 * 8, T=128, E=16, W=8, 38 KiB bank-conflict-free split smem):
// Phase 1 reg<4,7> -> Transpose 1 -> Phase 2 reg<4,3> -> Transpose 2 -> 2x reg_base<3> -> coalesced store.
template <int Sign>
void SYCL_FFT_SUFFIX(launch_single_pass_2048_wide)(
    sycl::queue& q, const SYCL_FFT_COMPLEX* in, SYCL_FFT_COMPLEX* out, size_t bs) {
    using Real = SYCL_FFT_REAL;
    using ComplexT = SYCL_FFT_COMPLEX;
    using SmemT = SYCL_FFT_SMEM_T;

    constexpr int N = 2048;
    constexpr int T = 128;
    constexpr int E = 16;
    constexpr int W = 8;
    constexpr int LogE = 4;
    constexpr int LogT = 7;
    constexpr int LogW = 3;
    constexpr int RowStride = 152; // 152 % 16 == 8 and 16*9 = 144 <= 152 -> 0 bank conflicts everywhere!
    constexpr int SmemElems = E * RowStride; // 2 * 16 * 152 * 8 B = 38 KiB

    if (bs == 0) return;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<SmemT, 1> smem_re(sycl::range<1>(SmemElems), h);
        sycl::local_accessor<SmemT, 1> smem_im(sycl::range<1>(SmemElems), h);

        h.parallel_for(
            sycl::nd_range<1>(bs * T, T),
            [=](sycl::nd_item<1> item) SYCL_FFT_LAMBDA_INLINE {
                const size_t batch = item.get_group_linear_id();
                const int t = static_cast<int>(item.get_local_linear_id());
                const ComplexT* g_in = in + batch * N;
                ComplexT* g_out = out + batch * N;

                Real r_re[E], r_im[E];
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    SYCL_FFT_LOAD_COMPLEX(g_in, t + M * T, r_re[M], r_im[M]);
                });

                SYCL_FFT_SUFFIX(dif_radix2_reg)<LogE, LogT, Sign>(r_re, r_im, t);

                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    smem_re[M * RowStride + t] = SYCL_FFT_TO_SMEM(r_re[M]);
                    smem_im[M * RowStride + t] = SYCL_FFT_TO_SMEM(r_im[M]);
                });
                sycl::group_barrier(item.get_group());

                const int sub = t >> LogW; // 0..15
                const int l = t & (W - 1); // 0..7
                const int row_base = sub * RowStride;
                const auto sg = item.get_sub_group();
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    r_re[M] = SYCL_FFT_FROM_SMEM(smem_re[row_base + M * W + l]);
                    r_im[M] = SYCL_FFT_FROM_SMEM(smem_im[row_base + M * W + l]);
                });

                SYCL_FFT_SUFFIX(dif_radix2_reg)<LogE, LogW, Sign>(r_re, r_im, l);

                // Transpose 2 via stride-9 padded rows (gcd(9, 16) = 1 -> 0 bank conflicts!)
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg);
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    smem_re[row_base + M * 9 + l] = SYCL_FFT_TO_SMEM(r_re[M]);
                    smem_im[row_base + M * 9 + l] = SYCL_FFT_TO_SMEM(r_im[M]);
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg);
                Real a_re[8], a_im[8], b_re[8], b_im[8];
                const int a_off = row_base + l * 9;
                const int b_off = row_base + (l + 8) * 9;
                unroll_for<8>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int K = k_ic.value;
                    a_re[K] = SYCL_FFT_FROM_SMEM(smem_re[a_off + K]);
                    a_im[K] = SYCL_FFT_FROM_SMEM(smem_im[a_off + K]);
                    b_re[K] = SYCL_FFT_FROM_SMEM(smem_re[b_off + K]);
                    b_im[K] = SYCL_FFT_FROM_SMEM(smem_im[b_off + K]);
                });

                // Phase 3: 100% warp-efficient register DIF (zero divergent subgroup cmuls!)
                SYCL_FFT_SUFFIX(dif_radix2_reg_base)<3, Sign>(a_re, a_im);
                SYCL_FFT_SUFFIX(dif_radix2_reg_base)<3, Sign>(b_re, b_im);

                // Coalesced global store via smem
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg);
                unroll_for<8>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int K = k_ic.value;
                    smem_re[a_off + K] = SYCL_FFT_TO_SMEM(a_re[K]);
                    smem_im[a_off + K] = SYCL_FFT_TO_SMEM(a_im[K]);
                    smem_re[b_off + K] = SYCL_FFT_TO_SMEM(b_re[K]);
                    smem_im[b_off + K] = SYCL_FFT_TO_SMEM(b_im[K]);
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg);

                const int out_base = sub * T + l;
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    const Real vr = SYCL_FFT_FROM_SMEM(smem_re[row_base + M * 9 + l]);
                    const Real vi = SYCL_FFT_FROM_SMEM(smem_im[row_base + M * 9 + l]);
                    SYCL_FFT_STORE_COMPLEX(g_out, out_base + M * W, vr, vi);
                });
            });
    });
}

// Single-Pass Wide N = 4096 (16 * 16 * 16, T=256, E=16, W=16, 32 KiB XOR-swizzled seq smem):
// Phase 1 reg<4,8> -> Transpose 1 -> Phase 2 reg<4,4> -> XOR Transpose 2 -> reg_base<4>.
template <int Sign>
void SYCL_FFT_SUFFIX(launch_single_pass_4096_wide)(
    sycl::queue& q, const SYCL_FFT_COMPLEX* in, SYCL_FFT_COMPLEX* out, size_t bs) {
    using Real = SYCL_FFT_REAL;
    using ComplexT = SYCL_FFT_COMPLEX;
    using SmemT = SYCL_FFT_SMEM_T;

    constexpr int N = 4096;
    constexpr int T = 256;
    constexpr int E = 16;
    constexpr int W = 16;
    constexpr int LogE = 4;
    constexpr int LogT = 8;
    constexpr int LogW = 4;
    constexpr int SmemElems = E * T; // 32 KiB

    if (bs == 0) return;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<SmemT, 1> smem(sycl::range<1>(SmemElems), h);

        h.parallel_for(
            sycl::nd_range<1>(bs * T, T),
            [=](sycl::nd_item<1> item) SYCL_FFT_LAMBDA_INLINE {
                const size_t batch = item.get_group_linear_id();
                const int t = static_cast<int>(item.get_local_linear_id());
                const ComplexT* g_in = in + batch * N;
                ComplexT* g_out = out + batch * N;

                Real r_re[E], r_im[E];
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    SYCL_FFT_LOAD_COMPLEX(g_in, t + M * T, r_re[M], r_im[M]);
                });

                SYCL_FFT_SUFFIX(dif_radix2_reg)<LogE, LogT, Sign>(r_re, r_im, t);

                const int sub = t >> LogW; // 0..15
                const int l = t & (W - 1); // 0..15
                const int sub_base = sub * T;
                const auto sg = item.get_sub_group();
                // Transpose 1 (seq real then imag)
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    smem[M * T + t] = SYCL_FFT_TO_SMEM(r_re[M]);
                });
                sycl::group_barrier(item.get_group());
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    r_re[M] = SYCL_FFT_FROM_SMEM(smem[sub_base + M * W + l]);
                });
                sycl::group_barrier(item.get_group());

                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    smem[M * T + t] = SYCL_FFT_TO_SMEM(r_im[M]);
                });
                sycl::group_barrier(item.get_group());
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    r_im[M] = SYCL_FFT_FROM_SMEM(smem[sub_base + M * W + l]);
                });

                SYCL_FFT_SUFFIX(dif_radix2_reg)<LogE, LogW, Sign>(r_re, r_im, l);

                // XOR-swizzled Transpose 2 (0 bank conflicts, 32 KiB seq smem)
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg);
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    smem[sub_base + M * W + (l ^ M)] = SYCL_FFT_TO_SMEM(r_re[M]);
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg);
                unroll_for<E>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int K = k_ic.value;
                    r_re[K] = SYCL_FFT_FROM_SMEM(smem[sub_base + l * W + (K ^ l)]);
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg);

                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    smem[sub_base + M * W + (l ^ M)] = SYCL_FFT_TO_SMEM(r_im[M]);
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg);
                unroll_for<E>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int K = k_ic.value;
                    r_im[K] = SYCL_FFT_FROM_SMEM(smem[sub_base + l * W + (K ^ l)]);
                });

                // Phase 3: 100% warp-efficient register DIF (zero divergent subgroup cmuls!)
                SYCL_FFT_SUFFIX(dif_radix2_reg_base)<LogE, Sign>(r_re, r_im);

                // Coalesced global store via XOR-swizzled seq smem (0 bank conflicts, stride-16 STG)
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg);
                unroll_for<E>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int K = k_ic.value;
                    smem[sub_base + l * W + (K ^ l)] = SYCL_FFT_TO_SMEM(r_re[K]);
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg);
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    r_re[M] = SYCL_FFT_FROM_SMEM(smem[sub_base + M * W + (l ^ M)]);
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg);

                unroll_for<E>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int K = k_ic.value;
                    smem[sub_base + l * W + (K ^ l)] = SYCL_FFT_TO_SMEM(r_im[K]);
                });
                SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg);
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    r_im[M] = SYCL_FFT_FROM_SMEM(smem[sub_base + M * W + (l ^ M)]);
                });

                const int out_base = sub_base + l;
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    SYCL_FFT_STORE_COMPLEX(g_out, out_base + M * W, r_re[M], r_im[M]);
                });
            });
    });
}

// Single-Pass Wide N = 8192 (32 * 16 * 16, T=256, E=32, 2 chunks of 16 in 32 KiB seq smem):
// Phase 1 reg<5,8> stashes chunk 1 (P=16..31) in g_out[4096..8191] (strictly thread-private
// addresses already loaded from g_in -> safe for both in-place and out-of-place, stays hot
// in L1/L2), freeing 64 live FP32 registers during chunk 0 to eliminate local-memory spills.
template <int Sign>
SYCL_FFT_FORCE_INLINE void SYCL_FFT_SUFFIX(finish_chunk_8192)(
    const sycl::nd_item<1>& item,
    const sycl::sub_group& sg,
    const sycl::local_accessor<SYCL_FFT_SMEM_T, 1>& smem,
    SYCL_FFT_REAL (&c_re)[16],
    SYCL_FFT_REAL (&c_im)[16],
    SYCL_FFT_COMPLEX* g_out,
    int chunk_idx,
    int t,
    int sub,
    int l) {
    using Real = SYCL_FFT_REAL;
    constexpr int T = 256;
    constexpr int W = 16;
    constexpr int Chunk = 16;
    const int sub_base = sub * T;

    // Transpose 1 (seq real then imag, CTA-wide)
    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int M = m_ic.value;
        smem[M * T + t] = SYCL_FFT_TO_SMEM(c_re[M]);
    });
    sycl::group_barrier(item.get_group());
    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int M = m_ic.value;
        c_re[M] = SYCL_FFT_FROM_SMEM(smem[sub_base + M * W + l]);
    });
    sycl::group_barrier(item.get_group());

    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int M = m_ic.value;
        smem[M * T + t] = SYCL_FFT_TO_SMEM(c_im[M]);
    });
    sycl::group_barrier(item.get_group());
    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int M = m_ic.value;
        c_im[M] = SYCL_FFT_FROM_SMEM(smem[sub_base + M * W + l]);
    });

    SYCL_FFT_SUFFIX(dif_radix2_reg)<4, 4, Sign>(c_re, c_im, l);

    // XOR-swizzled Transpose 2 (0 bank conflicts, intra-half-warp sub-group barriers)
    SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg);
    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int M = m_ic.value;
        smem[sub_base + M * W + (l ^ M)] = SYCL_FFT_TO_SMEM(c_re[M]);
    });
    SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg);
    unroll_for<Chunk>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int K = k_ic.value;
        c_re[K] = SYCL_FFT_FROM_SMEM(smem[sub_base + l * W + (K ^ l)]);
    });
    SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg);

    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int M = m_ic.value;
        smem[sub_base + M * W + (l ^ M)] = SYCL_FFT_TO_SMEM(c_im[M]);
    });
    SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg);
    unroll_for<Chunk>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int K = k_ic.value;
        c_im[K] = SYCL_FFT_FROM_SMEM(smem[sub_base + l * W + (K ^ l)]);
    });

    SYCL_FFT_SUFFIX(dif_radix2_reg_base)<4, Sign>(c_re, c_im);

    // Coalesced global store via XOR-swizzled seq smem (0 bank conflicts, stride-16 STG)
    SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg);
    unroll_for<Chunk>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int K = k_ic.value;
        smem[sub_base + l * W + (K ^ l)] = SYCL_FFT_TO_SMEM(c_re[K]);
    });
    SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg);
    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int M = m_ic.value;
        c_re[M] = SYCL_FFT_FROM_SMEM(smem[sub_base + M * W + (l ^ M)]);
    });
    SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg);

    unroll_for<Chunk>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int K = k_ic.value;
        smem[sub_base + l * W + (K ^ l)] = SYCL_FFT_TO_SMEM(c_im[K]);
    });
    SYCL_FFT_SUFFIX(sync_sub_tile)<W>(item, sg);
    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int M = m_ic.value;
        c_im[M] = SYCL_FFT_FROM_SMEM(smem[sub_base + M * W + (l ^ M)]);
    });

    const int out_base = (chunk_idx * Chunk + sub) * T + l;
    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int M = m_ic.value;
        SYCL_FFT_STORE_COMPLEX(g_out, out_base + M * W, c_re[M], c_im[M]);
    });
}

template <int Sign>
void SYCL_FFT_SUFFIX(launch_single_pass_8192)(
    sycl::queue& q, const SYCL_FFT_COMPLEX* in, SYCL_FFT_COMPLEX* out, size_t bs) {
    using Real = SYCL_FFT_REAL;
    using ComplexT = SYCL_FFT_COMPLEX;
    using SmemT = SYCL_FFT_SMEM_T;

    constexpr int N = 8192;
    constexpr int T = 256;
    constexpr int E = 32;
    constexpr int LogE = 5;
    constexpr int LogT = 8;
    constexpr int LogW = 4;
    constexpr int W = 16;
    constexpr int Chunk = 16;
    constexpr int SmemElems = Chunk * T; // 32 KiB seq smem

    if (bs == 0) return;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<SmemT, 1> smem(sycl::range<1>(SmemElems), h);

        h.parallel_for(
            sycl::nd_range<1>(bs * T, T),
            [=](sycl::nd_item<1> item) SYCL_FFT_LAMBDA_INLINE {
                const size_t batch = item.get_group_linear_id();
                const int t = static_cast<int>(item.get_local_linear_id());
                const ComplexT* g_in = in + batch * N;
                ComplexT* g_out = out + batch * N;

                const int sub = t >> LogW;
                const int l = t & (W - 1);
                const auto sg = item.get_sub_group();

                if constexpr (SYCL_FFT_STASH_8192 != 0) {
                    Real c_re[Chunk], c_im[Chunk];
                    {
                        Real r_re[E], r_im[E];
                        unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                            constexpr int M = m_ic.value;
                            SYCL_FFT_LOAD_COMPLEX(g_in, t + M * T, r_re[M], r_im[M]);
                        });

                        SYCL_FFT_SUFFIX(dif_radix2_reg_base)<LogE, Sign>(r_re, r_im);

                        // Stream chunk 0 (P=0..15) into c_re/c_im and stash chunk 1 (P=16..31)
                        // directly into g_out[4096..8191] at thread t's own already-loaded slots.
                        SYCL_FFT_SUFFIX(apply_bitrev_twiddles_with)<LogE, LogT, Sign>(
                            r_re, r_im, t, [&](auto p_ic, Real vr, Real vi) SYCL_FFT_LAMBDA_INLINE {
                                constexpr int P = p_ic.value;
                                if constexpr (P < Chunk) {
                                    c_re[P] = vr;
                                    c_im[P] = vi;
                                } else {
                                    SYCL_FFT_STORE_COMPLEX(g_out, (P * T) + t, vr, vi);
                                }
                            });
                    }

                    #pragma nounroll
                    for (int chunk_idx = 0; chunk_idx < 2; ++chunk_idx) {
                        if (chunk_idx > 0) {
                            sycl::group_barrier(item.get_group());
                            unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                                constexpr int M = m_ic.value;
                                SYCL_FFT_LOAD_COMPLEX(g_out, ((Chunk + M) * T) + t, c_re[M], c_im[M]);
                            });
                        }
                        SYCL_FFT_SUFFIX(finish_chunk_8192)<Sign>(
                            item, sg, smem, c_re, c_im, g_out, chunk_idx, t, sub, l);
                    }
                } else {
                    Real r_re[E], r_im[E];
                    unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                        constexpr int M = m_ic.value;
                        SYCL_FFT_LOAD_COMPLEX(g_in, t + M * T, r_re[M], r_im[M]);
                    });

                    SYCL_FFT_SUFFIX(dif_radix2_reg)<LogE, LogT, Sign>(r_re, r_im, t);

                    Real c_re[Chunk], c_im[Chunk];
                    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                        constexpr int M = m_ic.value;
                        c_re[M] = r_re[M];
                        c_im[M] = r_im[M];
                    });
                    SYCL_FFT_SUFFIX(finish_chunk_8192)<Sign>(
                        item, sg, smem, c_re, c_im, g_out, 0, t, sub, l);

                    sycl::group_barrier(item.get_group());
                    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                        constexpr int M = m_ic.value;
                        c_re[M] = r_re[Chunk + M];
                        c_im[M] = r_im[Chunk + M];
                    });
                    SYCL_FFT_SUFFIX(finish_chunk_8192)<Sign>(
                        item, sg, smem, c_re, c_im, g_out, 1, t, sub, l);
                }
            });
    });
}
#endif // SYCL_FFT_WIDE

// ============================================================================
// 2-Pass DIF Pass 0 (Register-only, ZERO shared memory, ZERO barriers):
//   N = N1 * N2, N1 = 2^LogN1 (8, 16, 32), N2 = 2^LogN2.
//   Runs plain N1-pt DIF in registers, multiplies slot P by W_N^{j * bitrev(P)},
//   and stores monotonically in ascending P = 0..N1-1 order.
// ============================================================================
template <int LogN1, int LogN2, int Sign>
void SYCL_FFT_SUFFIX(launch_pass0_reg)(
    sycl::queue& q,
    const SYCL_FFT_COMPLEX* in,
    SYCL_FFT_COMPLEX* tmp,
    size_t chunk_bs,
    size_t wg_size = 256) {
    using Real = SYCL_FFT_REAL;
    using ComplexT = SYCL_FFT_COMPLEX;

    constexpr int N1 = 1 << LogN1;
    constexpr int N2 = 1 << LogN2;
    constexpr int N = N1 * N2;

    if (chunk_bs == 0) return;
    const size_t total_threads = chunk_bs * N2;
    size_t wg = 1;
    const size_t cap = std::min(std::max<size_t>(1, wg_size), total_threads);
    while ((wg << 1) <= cap) wg <<= 1;
    while (wg > 1 && (total_threads % wg) != 0) wg >>= 1;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(total_threads, wg),
            [=](sycl::nd_item<1> item) SYCL_FFT_LAMBDA_INLINE {
                const size_t gid = item.get_global_linear_id();
                const size_t batch = gid >> LogN2;
                const int j = static_cast<int>(gid & (N2 - 1));

                const ComplexT* g_in = in + batch * N + j;
                ComplexT* g_tmp = tmp + batch * N + j;

                Real r_re[N1], r_im[N1];
                unroll_for<N1>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    SYCL_FFT_LOAD_COMPLEX(g_in, M * N2, r_re[M], r_im[M]);
                });

                SYCL_FFT_SUFFIX(dif_radix2_reg_base)<LogN1, Sign>(r_re, r_im);

                SYCL_FFT_SUFFIX(apply_bitrev_twiddles_with)<LogN1, LogN2, Sign>(
                    r_re, r_im, j, [&](auto p_ic, Real vr, Real vi) SYCL_FFT_LAMBDA_INLINE {
                        constexpr int P = p_ic.value;
                        SYCL_FFT_STORE_COMPLEX(g_tmp, P * N2, vr, vi);
                    });
            });
    });
}

// ============================================================================
// GPU Bit-Reversal Permutation Kernel (for correctness verification)
// ============================================================================
inline void SYCL_FFT_SUFFIX(launch_bit_reverse)(
    sycl::queue& q,
    const SYCL_FFT_COMPLEX* src,
    SYCL_FFT_COMPLEX* dst,
    int logN,
    size_t bs,
    size_t wg_size = 256) {
    using ComplexT = SYCL_FFT_COMPLEX;
    if (bs == 0) return;
    const size_t N = size_t{1} << logN;
    const size_t total = N * bs;
    size_t wg = 1;
    const size_t cap = std::min(std::max<size_t>(1, wg_size), total);
    while ((wg << 1) <= cap) wg <<= 1;
    while (wg > 1 && (total % wg) != 0) wg >>= 1;

    if (src == dst) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(total, wg), [=](sycl::nd_item<1> item) SYCL_FFT_LAMBDA_INLINE {
                const size_t gid = item.get_global_linear_id();
                const size_t batch = gid >> logN;
                const uint32_t idx = static_cast<uint32_t>(gid & (N - 1));
                const uint32_t rev = bit_reverse(idx, logN);
                if (idx < rev) {
                    ComplexT* base = dst + batch * N;
                    const ComplexT a = base[idx];
                    const ComplexT b = base[rev];
                    base[idx] = b;
                    base[rev] = a;
                }
            });
        });
    } else {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(total, wg), [=](sycl::nd_item<1> item) SYCL_FFT_LAMBDA_INLINE {
                const size_t gid = item.get_global_linear_id();
                const size_t batch = gid >> logN;
                const uint32_t idx = static_cast<uint32_t>(gid & (N - 1));
                const uint32_t rev = bit_reverse(idx, logN);
                dst[batch * N + idx] = src[batch * N + rev];
            });
        });
    }
}

// ============================================================================
// Adaptive Plan & Dispatcher (logN = 10..18)
//   Queries SYCL device config (L2 cache, compute units, local_mem_size,
//   max_work_group_size, sub_group_sizes) to derive single-pass vs two-pass
//   boundaries, wide vs narrow/seq kernel selection, work-group sizes, and
//   L2 chunk sizes — all overridable via FftLaunchConfig or execute_raw().
// ============================================================================
template <int Sign = -1>
class SYCL_FFT_SUFFIX(DifFftPlan) {
public:
    using ComplexT = SYCL_FFT_COMPLEX;
    using SmemT = SYCL_FFT_SMEM_T;

    SYCL_FFT_SUFFIX(DifFftPlan)(
        sycl::queue& q, int logN, size_t max_bs, const FftLaunchConfig& cfg = {})
        : q_(q), logN_(logN), N_(size_t{1} << logN),
          dev_cfg_(DeviceFftConfig::query(q.get_device())), launch_cfg_(cfg) {
        if (logN < 10 || logN > 18) {
            throw std::invalid_argument("Supported logN range is 10..18");
        }
        configure_plan(max_bs);
    }

    // Convenience constructor accepting an explicit L2 chunk size (0 = auto from device).
    SYCL_FFT_SUFFIX(DifFftPlan)(
        sycl::queue& q, int logN, size_t max_bs, size_t l2_chunk_bytes)
        : SYCL_FFT_SUFFIX(DifFftPlan)(q, logN, max_bs, make_chunk_config(l2_chunk_bytes)) {}

    ~SYCL_FFT_SUFFIX(DifFftPlan)() {
        if (scratch_) sycl::free(scratch_, q_);
    }

    SYCL_FFT_SUFFIX(DifFftPlan)(const SYCL_FFT_SUFFIX(DifFftPlan)&) = delete;
    SYCL_FFT_SUFFIX(DifFftPlan)& operator=(const SYCL_FFT_SUFFIX(DifFftPlan)&) = delete;

    // Bit-reversal-free (natural-to-reversed) DIF Radix-2 FFT.
    // Optional chunk_bs_override (> 0) caps the per-launch chunk batch size (<= chunk_bs()).
    void execute_raw(const ComplexT* in, ComplexT* out, size_t bs, size_t chunk_bs_override = 0) {
        if (bs == 0) return;
        if (!use_two_pass_) {
            dispatch_single_pass(logN_, in, out, bs);
            return;
        }
        const size_t eff_chunk = (chunk_bs_override > 0)
            ? std::min(chunk_bs_, chunk_bs_override)
            : chunk_bs_;
        dispatch_two_pass(in, out, bs, eff_chunk);
    }

    // Full natural-to-natural DIF Radix-2 FFT (for correctness verification)
    void execute(const ComplexT* in, ComplexT* out, size_t bs, size_t chunk_bs_override = 0) {
        if (bs == 0) return;
        execute_raw(in, out, bs, chunk_bs_override);
        SYCL_FFT_SUFFIX(launch_bit_reverse)(q_, out, out, logN_, bs, bitrev_wg_size_);
    }

    size_t chunk_bs() const { return chunk_bs_; }
    size_t chunk_bytes() const { return chunk_bs_ * N_ * sizeof(ComplexT); }
    size_t pass0_wg_size() const { return pass0_wg_size_; }
    size_t bitrev_wg_size() const { return bitrev_wg_size_; }
    int    two_pass_min_logN() const { return two_pass_min_logN_; }
    bool   is_two_pass() const { return use_two_pass_; }
    int    log_n1() const { return log_n1_; }
    int    log_n2() const { return log_n2_; }
    const DeviceFftConfig& device_config() const { return dev_cfg_; }
    const FftLaunchConfig& launch_config() const { return launch_cfg_; }

private:
    static FftLaunchConfig make_chunk_config(size_t l2_chunk_bytes) {
        FftLaunchConfig c{};
        c.l2_chunk_bytes = l2_chunk_bytes;
        return c;
    }

    bool fits_kernel(size_t req_wg, size_t req_smem, size_t req_sg = 1) const {
        return req_wg <= dev_cfg_.max_work_group_size &&
               req_smem <= dev_cfg_.local_mem_size &&
               req_sg <= dev_cfg_.min_sub_group_size;
    }

    bool can_single_pass(int n) const {
        constexpr size_t kSmemElem = sizeof(SmemT);
#if !SYCL_FFT_WIDE
        switch (n) {
            case 10:
                return fits_kernel(256, 2 * 2 * 8 * 144 * kSmemElem, 16) ||
                       fits_kernel(128, 2 * 1 * 8 * 144 * kSmemElem, 16) ||
                       fits_kernel(64,  2 * 1 * 16 * 80 * kSmemElem, 4);
            case 11:
                return fits_kernel(256, 2 * 8 * 256 * kSmemElem, 32) ||
                       fits_kernel(128, 2 * 16 * 144 * kSmemElem, 8);
            case 12:
                return fits_kernel(256, 2 * 16 * 272 * kSmemElem, 16) ||
                       fits_kernel(256, 16 * 256 * kSmemElem, 16);
            case 13:
                return fits_kernel(512, 16 * 512 * kSmemElem, 32);
            case 14:
                return fits_kernel(512, 16 * 512 * kSmemElem, 32);
            default:
                return false;
        }
#else
        switch (n) {
            case 10:
                return fits_kernel(64, 2 * 16 * 64 * kSmemElem, 4);
            case 11:
                return (launch_cfg_.allow_wide_kernels && fits_kernel(128, 2 * 16 * 152 * kSmemElem, 1)) ||
                       fits_kernel(128, 16 * 128 * kSmemElem, 8);
            case 12:
                return (launch_cfg_.allow_wide_kernels && fits_kernel(256, 16 * 256 * kSmemElem, 1)) ||
                       fits_kernel(256, 16 * 256 * kSmemElem, 16);
            case 13:
                return launch_cfg_.allow_wide_kernels && fits_kernel(256, 16 * 256 * kSmemElem, 1);
            default:
                return false;
        }
#endif
    }

    void configure_plan(size_t max_bs) {
        // Derive single-pass ceiling from device limits (local_mem_size, max_work_group_size, sub_group_sizes).
        int max_single = 10;
        for (int n = 10; n <= 14; ++n) {
            if (can_single_pass(n)) max_single = n;
        }
        two_pass_min_logN_ = (launch_cfg_.two_pass_min_logN > 0)
            ? launch_cfg_.two_pass_min_logN
            : (max_single + 1);

        use_two_pass_ = (logN_ >= two_pass_min_logN_) || !can_single_pass(logN_);

        // Bit-reversal work-group size (clamped to device limits and power-of-2 dividing N_).
        bitrev_wg_size_ = (launch_cfg_.bitrev_wg_size > 0)
            ? dev_cfg_.clamp_pow2_wg(N_, launch_cfg_.bitrev_wg_size)
            : dev_cfg_.default_wg_size(N_, 256);

        if (!use_two_pass_) {
            return;
        }

        // Choose (log_n1_, log_n2_) with 3 <= log_n1_ <= 5 and can_single_pass(log_n2_).
        auto valid_split = [&](int n1) -> bool {
            const int n2 = logN_ - n1;
            return n1 >= 3 && n1 <= 5 && n2 >= 10 && n2 <= 13 && can_single_pass(n2);
        };

        int chosen_n1 = 0;
        if (launch_cfg_.two_pass_log_n1 > 0 && valid_split(launch_cfg_.two_pass_log_n1)) {
            chosen_n1 = launch_cfg_.two_pass_log_n1;
        } else {
            // Precision- and hardware-aware preference:
            // - On DF (STASH_8192) at logN=17, prefer 32x4096_wide (<5,12>) over 16x8192 (<4,13>)
            //   because 4096_wide avoids 8192's higher register pressure and doubles Pass-1 WGs.
            // - First try Pass-1 sizes n2 >= 11 (higher throughput than n2=10), then allow n2=10.
            int order[3] = {4, 3, 5};
            if constexpr (SYCL_FFT_STASH_8192 != 0) {
                if (logN_ == 17) {
                    order[0] = 5; order[1] = 4; order[2] = 3;
                }
            }
            for (int min_n2 : {11, 10}) {
                for (int n1 : order) {
                    if (valid_split(n1) && (logN_ - n1) >= min_n2) {
                        chosen_n1 = n1;
                        break;
                    }
                }
                if (chosen_n1 != 0) break;
            }
        }
        if (chosen_n1 == 0) {
            throw std::runtime_error("Device capabilities cannot satisfy any 2-pass FFT split for this logN");
        }
        log_n1_ = chosen_n1;
        log_n2_ = logN_ - chosen_n1;

        const size_t n1 = size_t{1} << log_n1_;
        const size_t n2 = size_t{1} << log_n2_;
        const size_t fft_bytes = N_ * sizeof(ComplexT);
        const size_t l2_bytes = (launch_cfg_.l2_chunk_bytes > 0)
            ? launch_cfg_.l2_chunk_bytes
            : dev_cfg_.compute_l2_chunk_bytes(
                  fft_bytes,
                  n1,
                  SYCL_FFT_COMPUTE_HEAVY != 0,
                  launch_cfg_.l2_budget_num,
                  launch_cfg_.l2_budget_den,
                  launch_cfg_.min_waves_per_cu);

        size_t chunk_bs = std::max<size_t>(1, l2_bytes / fft_bytes);
        chunk_bs_ = std::min(max_bs, chunk_bs);
        if (chunk_bs_ == 0) chunk_bs_ = 1;

        // Register-aware cap for Pass 0 WG size (LogN1 == 5 on wide uses ~180 regs -> cap at 256).
        const size_t reg_wg_cap = (SYCL_FFT_WIDE && log_n1_ >= 5) ? 256 : 512;
        pass0_wg_size_ = (launch_cfg_.pass0_wg_size > 0)
            ? dev_cfg_.clamp_pow2_wg(n2, launch_cfg_.pass0_wg_size, reg_wg_cap)
            : dev_cfg_.default_wg_size(n2, std::min<size_t>(256, reg_wg_cap));

        scratch_elems_ = chunk_bs_ * N_;
        scratch_ = sycl::malloc_device<ComplexT>(scratch_elems_, q_);
        if (!scratch_) throw std::bad_alloc();
    }

    void dispatch_single_pass(int n, const ComplexT* in, ComplexT* out, size_t bs) {
        constexpr size_t kSmemElem = sizeof(SmemT);
#if !SYCL_FFT_WIDE
        switch (n) {
            case 10:
                if (fits_kernel(256, 2 * 2 * 8 * 144 * kSmemElem, 16)) {
                    SYCL_FFT_SUFFIX(launch_single_pass_split_smem)<10, 3, 2, Sign>(q_, in, out, bs);
                } else if (fits_kernel(128, 2 * 1 * 8 * 144 * kSmemElem, 16)) {
                    SYCL_FFT_SUFFIX(launch_single_pass_split_smem)<10, 3, 1, Sign>(q_, in, out, bs);
                } else {
                    SYCL_FFT_SUFFIX(launch_single_pass_split_smem)<10, 4, 1, Sign>(q_, in, out, bs);
                }
                break;
            case 11:
                if (fits_kernel(256, 2 * 8 * 256 * kSmemElem, 32)) {
                    SYCL_FFT_SUFFIX(launch_single_pass_split_smem)<11, 3, 1, Sign>(q_, in, out, bs);
                } else {
                    SYCL_FFT_SUFFIX(launch_single_pass_split_smem)<11, 4, 1, Sign>(q_, in, out, bs);
                }
                break;
            case 12:
                if (fits_kernel(256, 2 * 16 * 272 * kSmemElem, 16)) {
                    SYCL_FFT_SUFFIX(launch_single_pass_split_smem)<12, 4, 1, Sign>(q_, in, out, bs);
                } else {
                    SYCL_FFT_SUFFIX(launch_single_pass_seq_smem)<12, 4, Sign>(q_, in, out, bs);
                }
                break;
            case 13:
                SYCL_FFT_SUFFIX(launch_single_pass_seq_smem)<13, 4, Sign>(q_, in, out, bs);
                break;
            case 14:
                SYCL_FFT_SUFFIX(launch_single_pass_16384)<Sign>(q_, in, out, bs);
                break;
            default:
                throw std::invalid_argument("Unsupported single-pass logN");
        }
#else
        switch (n) {
            case 10:
                SYCL_FFT_SUFFIX(launch_single_pass_split_smem)<10, 4, 1, Sign>(q_, in, out, bs);
                break;
            case 11:
                if (launch_cfg_.allow_wide_kernels && fits_kernel(128, 2 * 16 * 152 * kSmemElem, 1)) {
                    SYCL_FFT_SUFFIX(launch_single_pass_2048_wide)<Sign>(q_, in, out, bs);
                } else {
                    SYCL_FFT_SUFFIX(launch_single_pass_seq_smem)<11, 4, Sign>(q_, in, out, bs);
                }
                break;
            case 12:
                if (launch_cfg_.allow_wide_kernels && fits_kernel(256, 16 * 256 * kSmemElem, 1)) {
                    SYCL_FFT_SUFFIX(launch_single_pass_4096_wide)<Sign>(q_, in, out, bs);
                } else {
                    SYCL_FFT_SUFFIX(launch_single_pass_seq_smem)<12, 4, Sign>(q_, in, out, bs);
                }
                break;
            case 13:
                SYCL_FFT_SUFFIX(launch_single_pass_8192)<Sign>(q_, in, out, bs);
                break;
            default:
                throw std::invalid_argument("Unsupported single-pass logN");
        }
#endif
    }

    template <int LogN1, int LogN2>
    void execute_two_pass(const ComplexT* in, ComplexT* out, size_t bs, size_t eff_chunk) {
        constexpr size_t N1 = size_t{1} << LogN1;
        size_t done = 0;
        while (done < bs) {
            const size_t cur = std::min(eff_chunk, bs - done);
            const ComplexT* c_in = in + done * N_;
            ComplexT* c_out = out + done * N_;

            SYCL_FFT_SUFFIX(launch_pass0_reg)<LogN1, LogN2, Sign>(
                q_, c_in, scratch_, cur, pass0_wg_size_);
            dispatch_single_pass(LogN2, scratch_, c_out, cur * N1);
            done += cur;
        }
    }

    void dispatch_two_pass(const ComplexT* in, ComplexT* out, size_t bs, size_t eff_chunk) {
        const int key = (log_n1_ << 8) | log_n2_;
        switch (key) {
            case (3 << 8) | 10: execute_two_pass<3, 10>(in, out, bs, eff_chunk); break;
            case (3 << 8) | 11: execute_two_pass<3, 11>(in, out, bs, eff_chunk); break;
            case (3 << 8) | 12: execute_two_pass<3, 12>(in, out, bs, eff_chunk); break;
            case (3 << 8) | 13: execute_two_pass<3, 13>(in, out, bs, eff_chunk); break;
            case (4 << 8) | 10: execute_two_pass<4, 10>(in, out, bs, eff_chunk); break;
            case (4 << 8) | 11: execute_two_pass<4, 11>(in, out, bs, eff_chunk); break;
            case (4 << 8) | 12: execute_two_pass<4, 12>(in, out, bs, eff_chunk); break;
            case (4 << 8) | 13: execute_two_pass<4, 13>(in, out, bs, eff_chunk); break;
            case (5 << 8) | 10: execute_two_pass<5, 10>(in, out, bs, eff_chunk); break;
            case (5 << 8) | 11: execute_two_pass<5, 11>(in, out, bs, eff_chunk); break;
            case (5 << 8) | 12: execute_two_pass<5, 12>(in, out, bs, eff_chunk); break;
            case (5 << 8) | 13: execute_two_pass<5, 13>(in, out, bs, eff_chunk); break;
            default:
                throw std::invalid_argument("Unsupported 2-pass FFT split");
        }
    }

    sycl::queue& q_;
    int logN_;
    size_t N_;
    DeviceFftConfig dev_cfg_;
    FftLaunchConfig launch_cfg_;
    int two_pass_min_logN_ = 0;
    bool use_two_pass_ = false;
    int log_n1_ = 0;
    int log_n2_ = 0;
    size_t pass0_wg_size_ = 256;
    size_t bitrev_wg_size_ = 256;
    size_t chunk_bs_ = 0;
    size_t scratch_elems_ = 0;
    ComplexT* scratch_ = nullptr;
};

} // namespace sycl_fft
