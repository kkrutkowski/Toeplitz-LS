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
//   FP64/DF:  logN = 10 (E=8, B=2, 32 KiB), 11 (E=8, B=1, 32 KiB)
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
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    smem_re[smem_base + M * RowStride + t] = SYCL_FFT_TO_SMEM(r_re[M]);
                    smem_im[smem_base + M * RowStride + t] = SYCL_FFT_TO_SMEM(r_im[M]);
                });

                sycl::group_barrier(item.get_group());

                const int sub = t >> LogW;
                const int l = t & (W - 1);
                const int row_off = smem_base + sub * RowStride + l;
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    r_re[M] = SYCL_FFT_FROM_SMEM(smem_re[row_off + M * W]);
                    r_im[M] = SYCL_FFT_FROM_SMEM(smem_im[row_off + M * W]);
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
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    r_re[M] = SYCL_FFT_FROM_SMEM(smem_re[row_base + M * W + l]);
                    r_im[M] = SYCL_FFT_FROM_SMEM(smem_im[row_base + M * W + l]);
                });

                SYCL_FFT_SUFFIX(dif_radix2_reg)<LogE, LogW, Sign>(r_re, r_im, l);

                // Transpose 2 via stride-9 padded rows (gcd(9, 16) = 1 -> 0 bank conflicts!)
                sycl::group_barrier(item.get_group());
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    smem_re[row_base + M * 9 + l] = SYCL_FFT_TO_SMEM(r_re[M]);
                    smem_im[row_base + M * 9 + l] = SYCL_FFT_TO_SMEM(r_im[M]);
                });
                sycl::group_barrier(item.get_group());

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
                sycl::group_barrier(item.get_group());
                unroll_for<8>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int K = k_ic.value;
                    smem_re[a_off + K] = SYCL_FFT_TO_SMEM(a_re[K]);
                    smem_im[a_off + K] = SYCL_FFT_TO_SMEM(a_im[K]);
                    smem_re[b_off + K] = SYCL_FFT_TO_SMEM(b_re[K]);
                    smem_im[b_off + K] = SYCL_FFT_TO_SMEM(b_im[K]);
                });
                sycl::group_barrier(item.get_group());

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
                sycl::group_barrier(item.get_group());
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    smem[sub_base + M * W + (l ^ M)] = SYCL_FFT_TO_SMEM(r_re[M]);
                });
                sycl::group_barrier(item.get_group());
                unroll_for<E>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int K = k_ic.value;
                    r_re[K] = SYCL_FFT_FROM_SMEM(smem[sub_base + l * W + (K ^ l)]);
                });
                sycl::group_barrier(item.get_group());

                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    smem[sub_base + M * W + (l ^ M)] = SYCL_FFT_TO_SMEM(r_im[M]);
                });
                sycl::group_barrier(item.get_group());
                unroll_for<E>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int K = k_ic.value;
                    r_im[K] = SYCL_FFT_FROM_SMEM(smem[sub_base + l * W + (K ^ l)]);
                });

                // Phase 3: 100% warp-efficient register DIF (zero divergent subgroup cmuls!)
                SYCL_FFT_SUFFIX(dif_radix2_reg_base)<LogE, Sign>(r_re, r_im);

                // Coalesced global store via XOR-swizzled seq smem (0 bank conflicts, stride-16 STG)
                sycl::group_barrier(item.get_group());
                unroll_for<E>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int K = k_ic.value;
                    smem[sub_base + l * W + (K ^ l)] = SYCL_FFT_TO_SMEM(r_re[K]);
                });
                sycl::group_barrier(item.get_group());
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    r_re[M] = SYCL_FFT_FROM_SMEM(smem[sub_base + M * W + (l ^ M)]);
                });
                sycl::group_barrier(item.get_group());

                unroll_for<E>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int K = k_ic.value;
                    smem[sub_base + l * W + (K ^ l)] = SYCL_FFT_TO_SMEM(r_im[K]);
                });
                sycl::group_barrier(item.get_group());
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
// Phase 1 reg<5,8> -> per chunk: Transpose 1 -> reg<4,4> -> XOR Transpose 2 -> reg_base<4>.
template <int C, int Sign>
SYCL_FFT_FORCE_INLINE void SYCL_FFT_SUFFIX(finish_chunk_8192)(
    const sycl::nd_item<1>& item,
    const sycl::local_accessor<SYCL_FFT_SMEM_T, 1>& smem,
    const SYCL_FFT_REAL (&r_re)[32],
    const SYCL_FFT_REAL (&r_im)[32],
    SYCL_FFT_COMPLEX* g_out,
    int t,
    int sub,
    int l) {
    using Real = SYCL_FFT_REAL;
    constexpr int T = 256;
    constexpr int W = 16;
    constexpr int Chunk = 16;
    const int sub_base = sub * T;

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
        c_re[M] = SYCL_FFT_FROM_SMEM(smem[sub_base + M * W + l]);
    });
    sycl::group_barrier(item.get_group());

    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int M = m_ic.value;
        smem[M * T + t] = SYCL_FFT_TO_SMEM(r_im[C * Chunk + M]);
    });
    sycl::group_barrier(item.get_group());
    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int M = m_ic.value;
        c_im[M] = SYCL_FFT_FROM_SMEM(smem[sub_base + M * W + l]);
    });

    SYCL_FFT_SUFFIX(dif_radix2_reg)<4, 4, Sign>(c_re, c_im, l);

    // XOR-swizzled Transpose 2 (0 bank conflicts) -> 100%-warp-efficient reg_base<4>
    sycl::group_barrier(item.get_group());
    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int M = m_ic.value;
        smem[sub_base + M * W + (l ^ M)] = SYCL_FFT_TO_SMEM(c_re[M]);
    });
    sycl::group_barrier(item.get_group());
    unroll_for<Chunk>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int K = k_ic.value;
        c_re[K] = SYCL_FFT_FROM_SMEM(smem[sub_base + l * W + (K ^ l)]);
    });
    sycl::group_barrier(item.get_group());

    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int M = m_ic.value;
        smem[sub_base + M * W + (l ^ M)] = SYCL_FFT_TO_SMEM(c_im[M]);
    });
    sycl::group_barrier(item.get_group());
    unroll_for<Chunk>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int K = k_ic.value;
        c_im[K] = SYCL_FFT_FROM_SMEM(smem[sub_base + l * W + (K ^ l)]);
    });

    SYCL_FFT_SUFFIX(dif_radix2_reg_base)<4, Sign>(c_re, c_im);

    // Coalesced global store via XOR-swizzled seq smem (0 bank conflicts, stride-16 STG)
    sycl::group_barrier(item.get_group());
    unroll_for<Chunk>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int K = k_ic.value;
        smem[sub_base + l * W + (K ^ l)] = SYCL_FFT_TO_SMEM(c_re[K]);
    });
    sycl::group_barrier(item.get_group());
    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int M = m_ic.value;
        c_re[M] = SYCL_FFT_FROM_SMEM(smem[sub_base + M * W + (l ^ M)]);
    });
    sycl::group_barrier(item.get_group());

    unroll_for<Chunk>([&](auto k_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int K = k_ic.value;
        smem[sub_base + l * W + (K ^ l)] = SYCL_FFT_TO_SMEM(c_im[K]);
    });
    sycl::group_barrier(item.get_group());
    unroll_for<Chunk>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
        constexpr int M = m_ic.value;
        c_im[M] = SYCL_FFT_FROM_SMEM(smem[sub_base + M * W + (l ^ M)]);
    });

    const int out_base = (C * Chunk + sub) * T + l;
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

                Real r_re[E], r_im[E];
                unroll_for<E>([&](auto m_ic) SYCL_FFT_LAMBDA_INLINE {
                    constexpr int M = m_ic.value;
                    SYCL_FFT_LOAD_COMPLEX(g_in, t + M * T, r_re[M], r_im[M]);
                });

                SYCL_FFT_SUFFIX(dif_radix2_reg)<LogE, LogT, Sign>(r_re, r_im, t);

                const int sub = t >> LogW;
                const int l = t & (W - 1);

                SYCL_FFT_SUFFIX(finish_chunk_8192)<0, Sign>(item, smem, r_re, r_im, g_out, t, sub, l);
                SYCL_FFT_SUFFIX(finish_chunk_8192)<1, Sign>(item, smem, r_re, r_im, g_out, t, sub, l);
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
    sycl::queue& q, const SYCL_FFT_COMPLEX* in, SYCL_FFT_COMPLEX* tmp, size_t chunk_bs) {
    using Real = SYCL_FFT_REAL;
    using ComplexT = SYCL_FFT_COMPLEX;

    constexpr int N1 = 1 << LogN1;
    constexpr int N2 = 1 << LogN2;
    constexpr int N = N1 * N2;
    constexpr int WG = 256;

    if (chunk_bs == 0) return;
    const size_t total_threads = chunk_bs * N2;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(total_threads, WG),
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
    sycl::queue& q, const SYCL_FFT_COMPLEX* src, SYCL_FFT_COMPLEX* dst, int logN, size_t bs) {
    using ComplexT = SYCL_FFT_COMPLEX;
    if (bs == 0) return;
    const size_t N = size_t{1} << logN;
    const size_t total = N * bs;
    constexpr size_t WG = 256;

    if (src == dst) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(total, WG), [=](sycl::nd_item<1> item) SYCL_FFT_LAMBDA_INLINE {
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
            h.parallel_for(sycl::nd_range<1>(total, WG), [=](sycl::nd_item<1> item) SYCL_FFT_LAMBDA_INLINE {
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
// Plan & Dispatcher (logN = 10..18)
//   FP32 (!SYCL_FFT_WIDE):   10..14 single-pass; 15..18 2-pass L2-chunked (6 MiB default)
//   FP64/DF (SYCL_FFT_WIDE): 10..13 single-pass; 14..18 2-pass L2-chunked (16 MiB default)
// ============================================================================
template <int Sign = -1>
class SYCL_FFT_SUFFIX(DifFftPlan) {
public:
    using ComplexT = SYCL_FFT_COMPLEX;
    static constexpr int kTwoPassMinLogN = SYCL_FFT_WIDE ? 14 : 15;
    static constexpr size_t kDefaultL2ChunkBytes = (SYCL_FFT_WIDE ? 16 : 6) * size_t{1024} * 1024;

    SYCL_FFT_SUFFIX(DifFftPlan)(
        sycl::queue& q, int logN, size_t max_bs, size_t l2_chunk_bytes = kDefaultL2ChunkBytes)
        : q_(q), logN_(logN), N_(size_t{1} << logN) {
        if (logN < 10 || logN > 18) {
            throw std::invalid_argument("Supported logN range is 10..18");
        }
        if (logN >= kTwoPassMinLogN) {
            const size_t fft_bytes = N_ * sizeof(ComplexT);
            size_t chunk_bs = std::max<size_t>(1, l2_chunk_bytes / fft_bytes);
            chunk_bs_ = std::min(max_bs, chunk_bs);
            if (chunk_bs_ == 0) chunk_bs_ = 1;
            scratch_elems_ = chunk_bs_ * N_;
            scratch_ = sycl::malloc_device<ComplexT>(scratch_elems_, q_);
            if (!scratch_) throw std::bad_alloc();
        }
    }

    ~SYCL_FFT_SUFFIX(DifFftPlan)() {
        if (scratch_) sycl::free(scratch_, q_);
    }

    SYCL_FFT_SUFFIX(DifFftPlan)(const SYCL_FFT_SUFFIX(DifFftPlan)&) = delete;
    SYCL_FFT_SUFFIX(DifFftPlan)& operator=(const SYCL_FFT_SUFFIX(DifFftPlan)&) = delete;

    // Bit-reversal-free (natural-to-reversed) DIF Radix-2 FFT
    void execute_raw(const ComplexT* in, ComplexT* out, size_t bs) {
        if (bs == 0) return;
#if !SYCL_FFT_WIDE
        switch (logN_) {
            case 10:
                SYCL_FFT_SUFFIX(launch_single_pass_split_smem)<10, 3, 2, Sign>(q_, in, out, bs);
                break;
            case 11:
                SYCL_FFT_SUFFIX(launch_single_pass_split_smem)<11, 3, 1, Sign>(q_, in, out, bs);
                break;
            case 12:
                SYCL_FFT_SUFFIX(launch_single_pass_split_smem)<12, 4, 1, Sign>(q_, in, out, bs);
                break;
            case 13:
                SYCL_FFT_SUFFIX(launch_single_pass_seq_smem)<13, 4, Sign>(q_, in, out, bs);
                break;
            case 14:
                SYCL_FFT_SUFFIX(launch_single_pass_16384)<Sign>(q_, in, out, bs);
                break;
            case 15:
                execute_two_pass<4, 11>(in, out, bs);
                break;
            case 16:
                execute_two_pass<4, 12>(in, out, bs);
                break;
            case 17:
                execute_two_pass<4, 13>(in, out, bs);
                break;
            case 18:
                execute_two_pass<5, 13>(in, out, bs);
                break;
            default:
                throw std::invalid_argument("Unsupported logN");
        }
#else
        switch (logN_) {
            case 10:
                SYCL_FFT_SUFFIX(launch_single_pass_split_smem)<10, 4, 1, Sign>(q_, in, out, bs);
                break;
            case 11:
                SYCL_FFT_SUFFIX(launch_single_pass_2048_wide)<Sign>(q_, in, out, bs);
                break;
            case 12:
                SYCL_FFT_SUFFIX(launch_single_pass_4096_wide)<Sign>(q_, in, out, bs);
                break;
            case 13:
                SYCL_FFT_SUFFIX(launch_single_pass_8192)<Sign>(q_, in, out, bs);
                break;
            case 14:
                execute_two_pass<3, 11>(in, out, bs);
                break;
            case 15:
                execute_two_pass<4, 11>(in, out, bs);
                break;
            case 16:
                execute_two_pass<4, 12>(in, out, bs);
                break;
            case 17:
                execute_two_pass<4, 13>(in, out, bs);
                break;
            case 18:
                execute_two_pass<5, 13>(in, out, bs);
                break;
            default:
                throw std::invalid_argument("Unsupported logN");
        }
#endif
    }

    // Full natural-to-natural DIF Radix-2 FFT (for correctness verification)
    void execute(const ComplexT* in, ComplexT* out, size_t bs) {
        if (bs == 0) return;
        execute_raw(in, out, bs);
        SYCL_FFT_SUFFIX(launch_bit_reverse)(q_, out, out, logN_, bs);
    }

    size_t chunk_bs() const { return chunk_bs_; }

private:
    template <int LogN1, int LogN2>
    void execute_two_pass(const ComplexT* in, ComplexT* out, size_t bs) {
        constexpr size_t N1 = size_t{1} << LogN1;
        size_t done = 0;
        while (done < bs) {
            const size_t cur = std::min(chunk_bs_, bs - done);
            const ComplexT* c_in = in + done * N_;
            ComplexT* c_out = out + done * N_;

            SYCL_FFT_SUFFIX(launch_pass0_reg)<LogN1, LogN2, Sign>(q_, c_in, scratch_, cur);

#if !SYCL_FFT_WIDE
            if constexpr (LogN2 == 11) {
                SYCL_FFT_SUFFIX(launch_single_pass_split_smem)<11, 3, 1, Sign>(q_, scratch_, c_out, cur * N1);
            } else if constexpr (LogN2 == 12) {
                SYCL_FFT_SUFFIX(launch_single_pass_split_smem)<12, 4, 1, Sign>(q_, scratch_, c_out, cur * N1);
            } else if constexpr (LogN2 == 13) {
                SYCL_FFT_SUFFIX(launch_single_pass_seq_smem)<13, 4, Sign>(q_, scratch_, c_out, cur * N1);
            }
#else
            if constexpr (LogN2 == 11) {
                SYCL_FFT_SUFFIX(launch_single_pass_2048_wide)<Sign>(q_, scratch_, c_out, cur * N1);
            } else if constexpr (LogN2 == 12) {
                SYCL_FFT_SUFFIX(launch_single_pass_4096_wide)<Sign>(q_, scratch_, c_out, cur * N1);
            } else if constexpr (LogN2 == 13) {
                SYCL_FFT_SUFFIX(launch_single_pass_8192)<Sign>(q_, scratch_, c_out, cur * N1);
            }
#endif
            done += cur;
        }
    }

    sycl::queue& q_;
    int logN_;
    size_t N_;
    size_t chunk_bs_ = 0;
    size_t scratch_elems_ = 0;
    ComplexT* scratch_ = nullptr;
};

} // namespace sycl_fft
