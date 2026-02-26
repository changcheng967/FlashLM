#include <arm_neon.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

/* ===== FAST EXP (NEON) ===== */
static inline float32x4_t fast_exp_neon(float32x4_t x) {
    x = vmaxq_f32(x, vdupq_n_f32(-88.0f));
    x = vminq_f32(x, vdupq_n_f32(88.0f));
    float32x4_t log2e = vdupq_n_f32(1.44269504f);
    float32x4_t t = vmulq_f32(x, log2e);
    int32x4_t n = vcvtq_s32_f32(t);
    float32x4_t nf = vcvtq_f32_s32(n);
    uint32x4_t mask = vcgtq_f32(nf, t);
    n = vsubq_s32(n, vandq_s32(vreinterpretq_s32_u32(mask), vdupq_n_s32(1)));
    nf = vcvtq_f32_s32(n);
    float32x4_t f = vsubq_f32(t, nf);
    float32x4_t c1 = vdupq_n_f32(0.693147180f);
    float32x4_t c2 = vdupq_n_f32(0.240226507f);
    float32x4_t c3 = vdupq_n_f32(0.055504109f);
    float32x4_t p = vmlaq_f32(c2, c3, f);
    p = vmlaq_f32(c1, p, f);
    p = vmlaq_f32(vdupq_n_f32(1.0f), p, f);
    int32x4_t exp_bits = vshlq_n_s32(n, 23);
    float32x4_t pow2n = vreinterpretq_f32_s32(
        vaddq_s32(vreinterpretq_s32_f32(vdupq_n_f32(1.0f)), exp_bits));
    return vmulq_f32(p, pow2n);
}

/* ===== TERNARY MATMUL (NEON popcount) ===== */
void ternary_matmul(
    const uint8_t* X_val, const uint8_t* X_sign,
    const uint8_t* W_val, const uint8_t* W_sign,
    int32_t* C, int M, int N, int KB
) {
    int MN = M * N;
    #pragma omp parallel for schedule(static, 256)
    for (int idx = 0; idx < MN; idx++) {
        int i = idx / N;
        int j = idx % N;
        const uint8_t* xi_v = X_val  + (long)i * KB;
        const uint8_t* xi_s = X_sign + (long)i * KB;
        const uint8_t* wj_v = W_val  + (long)j * KB;
        const uint8_t* wj_s = W_sign + (long)j * KB;
        int32_t ps = 0, ns = 0;
        int k = 0;
        for (; k + 16 <= KB; k += 16) {
            uint8x16_t xv = vld1q_u8(xi_v + k);
            uint8x16_t xs = vld1q_u8(xi_s + k);
            uint8x16_t wv = vld1q_u8(wj_v + k);
            uint8x16_t ws = vld1q_u8(wj_s + k);
            uint8x16_t bv = vandq_u8(xv, wv);
            uint8x16_t bs = vandq_u8(veorq_u8(xs, ws), bv);
            ps += vaddvq_u16(vpaddlq_u8(vcntq_u8(bv)));
            ns += vaddvq_u16(vpaddlq_u8(vcntq_u8(bs)));
        }
        for (; k < KB; k++) {
            uint8_t bv = xi_v[k] & wj_v[k];
            uint8_t bs = (xi_s[k] ^ wj_s[k]) & bv;
            ps += __builtin_popcount(bv);
            ns += __builtin_popcount(bs);
        }
        C[idx] = ps - 2 * ns;
    }
}

/* ===== TERNARY WEIGHT GRADIENT (scatter-add) with NEON ===== */
void ternary_transpose_matmul_f32(
    const uint8_t* X_val, const uint8_t* X_sign,
    const float* grad, float* dW,
    int M, int K, int N
) {
    int KB = (K + 7) / 8;
    long KN = (long)K * N;
    memset(dW, 0, KN * sizeof(float));

    int nthreads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
    }

    float** local_dW = (float**)malloc(nthreads * sizeof(float*));
    for (int t = 0; t < nthreads; t++) {
        local_dW[t] = (float*)calloc(KN, sizeof(float));
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float* my_dW = local_dW[tid];

        #pragma omp for schedule(static)
        for (int i = 0; i < M; i++) {
            const uint8_t* vi = X_val  + (long)i * KB;
            const uint8_t* si = X_sign + (long)i * KB;
            const float* gi = grad + (long)i * N;

            for (int k = 0; k < K; k++) {
                int byte_idx = k / 8;
                int bit_idx = 7 - (k % 8);
                uint8_t mask = 1 << bit_idx;
                if (!(vi[byte_idx] & mask)) continue;

                float* dw_row = my_dW + (long)k * N;
                if (si[byte_idx] & mask) {
                    /* NEON vectorized subtract */
                    int n = 0;
                    for (; n + 4 <= N; n += 4) {
                        float32x4_t dw = vld1q_f32(dw_row + n);
                        float32x4_t gv = vld1q_f32(gi + n);
                        vst1q_f32(dw_row + n, vsubq_f32(dw, gv));
                    }
                    for (; n < N; n++) dw_row[n] -= gi[n];
                } else {
                    /* NEON vectorized add */
                    int n = 0;
                    for (; n + 4 <= N; n += 4) {
                        float32x4_t dw = vld1q_f32(dw_row + n);
                        float32x4_t gv = vld1q_f32(gi + n);
                        vst1q_f32(dw_row + n, vaddq_f32(dw, gv));
                    }
                    for (; n < N; n++) dw_row[n] += gi[n];
                }
            }
        }
    }

    #pragma omp parallel for schedule(static, 1024)
    for (long idx = 0; idx < KN; idx++) {
        float sum = 0.0f;
        for (int t = 0; t < nthreads; t++) sum += local_dW[t][idx];
        dW[idx] = sum;
    }

    for (int t = 0; t < nthreads; t++) free(local_dW[t]);
    free(local_dW);
}

/* ===== SiLU (NEON) - single exp, two Newton steps ===== */
void silu_f32(float* x, int n) {
    #pragma omp parallel for schedule(static, 4096)
    for (int i = 0; i < n - 3; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t neg_v = vnegq_f32(v);
        float32x4_t e = fast_exp_neon(neg_v);
        float32x4_t denom = vaddq_f32(vdupq_n_f32(1.0f), e);
        float32x4_t inv = vrecpeq_f32(denom);
        inv = vmulq_f32(inv, vrecpsq_f32(denom, inv));  /* Newton 1 */
        inv = vmulq_f32(inv, vrecpsq_f32(denom, inv));  /* Newton 2 (reuse denom, no re-exp) */
        vst1q_f32(x + i, vmulq_f32(v, inv));
    }
    for (int i = (n & ~3); i < n; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

/* ===== SiLU BACKWARD (NEON) ===== */
void silu_bwd_f32(const float* x, const float* grad_out, float* grad_in, int n) {
    #pragma omp parallel for schedule(static, 4096)
    for (int i = 0; i < n - 3; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t go = vld1q_f32(grad_out + i);
        float32x4_t neg_v = vnegq_f32(v);
        float32x4_t e = fast_exp_neon(neg_v);
        float32x4_t one = vdupq_n_f32(1.0f);
        float32x4_t denom = vaddq_f32(one, e);
        float32x4_t inv = vrecpeq_f32(denom);
        inv = vmulq_f32(inv, vrecpsq_f32(denom, inv));
        inv = vmulq_f32(inv, vrecpsq_f32(denom, inv));
        float32x4_t one_minus_sig = vsubq_f32(one, inv);
        float32x4_t dsilu = vmulq_f32(inv, vmlaq_f32(one, v, one_minus_sig));
        vst1q_f32(grad_in + i, vmulq_f32(go, dsilu));
    }
    for (int i = (n & ~3); i < n; i++) {
        float s = 1.0f / (1.0f + expf(-x[i]));
        grad_in[i] = grad_out[i] * (s + x[i] * s * (1.0f - s));
    }
}

/* ===== RMSNORM FORWARD (NEON) ===== */
void rmsnorm_f32(const float* x, const float* gamma, float* out, int M, int D) {
    int D4 = D & ~3;
    #pragma omp parallel for schedule(static, 32)
    for (int i = 0; i < M; i++) {
        const float* xi = x + (long)i * D;
        float* oi = out + (long)i * D;

        /* NEON vectorized sum of squares */
        float32x4_t vsum = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j < D4; j += 4) {
            float32x4_t v = vld1q_f32(xi + j);
            vsum = vmlaq_f32(vsum, v, v);
        }
        float sum = vaddvq_f32(vsum);
        for (; j < D; j++) sum += xi[j] * xi[j];

        float scale = 1.0f / sqrtf(sum / D + 1e-6f);
        float32x4_t vs = vdupq_n_f32(scale);

        /* NEON vectorized scale and gamma multiply */
        j = 0;
        for (; j < D4; j += 4) {
            float32x4_t v = vld1q_f32(xi + j);
            float32x4_t g = vld1q_f32(gamma + j);
            vst1q_f32(oi + j, vmulq_f32(vmulq_f32(v, vs), g));
        }
        for (; j < D; j++) oi[j] = xi[j] * scale * gamma[j];
    }
}

/* ===== RMSNORM BACKWARD (NEON) ===== */
void rmsnorm_bwd_f32(const float* x, const float* gamma, const float* grad_out,
                      float* grad_gamma, float* grad_in, int M, int D) {
    /* Zero grad_gamma */
    for (int j = 0; j < D; j++) grad_gamma[j] = 0.0f;

    int D4 = D & ~3;

    int nthreads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
    }

    float** local_gg = (float**)malloc(nthreads * sizeof(float*));
    for (int t = 0; t < nthreads; t++) {
        local_gg[t] = (float*)calloc(D, sizeof(float));
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float* my_gg = local_gg[tid];

        #pragma omp for schedule(static, 32)
        for (int i = 0; i < M; i++) {
            const float* xi = x + (long)i * D;
            const float* go = grad_out + (long)i * D;
            float* gi = grad_in + (long)i * D;

            /* NEON vectorized sum of squares */
            float32x4_t vsum = vdupq_n_f32(0.0f);
            int j = 0;
            for (; j < D4; j += 4) {
                float32x4_t v = vld1q_f32(xi + j);
                vsum = vmlaq_f32(vsum, v, v);
            }
            float sum = vaddvq_f32(vsum);
            for (; j < D; j++) sum += xi[j] * xi[j];

            float rms2 = sum / D + 1e-6f;
            float inv_rms = 1.0f / sqrtf(rms2);

            /* NEON vectorized dot product: go * gamma * xi */
            float32x4_t vdot = vdupq_n_f32(0.0f);
            j = 0;
            for (; j < D4; j += 4) {
                float32x4_t g = vld1q_f32(gamma + j);
                float32x4_t go_v = vld1q_f32(go + j);
                float32x4_t xi_v = vld1q_f32(xi + j);
                vdot = vmlaq_f32(vdot, vmulq_f32(go_v, g), xi_v);
            }
            float dot = vaddvq_f32(vdot);
            for (; j < D; j++) dot += go[j] * gamma[j] * xi[j];

            dot *= inv_rms * inv_rms / D;

            /* NEON vectorized gradient computation */
            float32x4_t vinv_rms = vdupq_n_f32(inv_rms);
            float32x4_t vdot4 = vdupq_n_f32(dot);
            j = 0;
            for (; j < D4; j += 4) {
                float32x4_t go_v = vld1q_f32(go + j);
                float32x4_t g = vld1q_f32(gamma + j);
                float32x4_t xi_v = vld1q_f32(xi + j);
                /* gi = (go * gamma - xi * dot) * inv_rms */
                float32x4_t term1 = vmulq_f32(go_v, g);
                float32x4_t term2 = vmulq_f32(xi_v, vdot4);
                float32x4_t gi_v = vmulq_f32(vsubq_f32(term1, term2), vinv_rms);
                vst1q_f32(gi + j, gi_v);
                /* my_gg += go * xi * inv_rms */
                float32x4_t gg_v = vld1q_f32(my_gg + j);
                gg_v = vmlaq_f32(gg_v, vmulq_f32(go_v, xi_v), vinv_rms);
                vst1q_f32(my_gg + j, gg_v);
            }
            for (; j < D; j++) {
                gi[j] = (go[j] * gamma[j] - xi[j] * dot) * inv_rms;
                my_gg[j] += go[j] * xi[j] * inv_rms;
            }
        }
    }

    /* Reduce grad_gamma */
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < D; j++) {
        float sum = 0.0f;
        for (int t = 0; t < nthreads; t++) sum += local_gg[t][j];
        grad_gamma[j] = sum;
    }

    for (int t = 0; t < nthreads; t++) free(local_gg[t]);
    free(local_gg);
}

/* ===== REQUANTIZE: float32 -> packed ternary (NEON) ===== */
void requantize_f32(const float* x, uint8_t* out_val, uint8_t* out_sign, int M, int D) {
    int KB = (D + 7) / 8;
    int D4 = D & ~3;
    #pragma omp parallel for schedule(static, 32)
    for (int i = 0; i < M; i++) {
        const float* xi = x + (long)i * D;
        uint8_t* ov = out_val + (long)i * KB;
        uint8_t* os = out_sign + (long)i * KB;

        /* NEON vectorized fabsf sum */
        float32x4_t vabs_sum = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j < D4; j += 4)
            vabs_sum = vaddq_f32(vabs_sum, vabsq_f32(vld1q_f32(xi + j)));
        float sum = vaddvq_f32(vabs_sum);
        for (; j < D; j++) sum += fabsf(xi[j]);

        float thresh = sum / D * 0.6745f;
        memset(ov, 0, KB);
        memset(os, 0, KB);
        for (int j = 0; j < D; j++) {
            int bi = j / 8, bt = 7 - (j % 8);
            if (xi[j] > thresh) {
                ov[bi] |= (1 << bt);
            } else if (xi[j] < -thresh) {
                ov[bi] |= (1 << bt);
                os[bi] |= (1 << bt);
            }
        }
    }
}

/* ===== CROSS-ENTROPY FORWARD+BACKWARD (NEON) ===== */
float cross_entropy_fwd_bwd(const float* logits, const int32_t* targets,
                             float* grad, int M, int V) {
    double total_loss = 0.0;
    int V4 = V & ~3;

    #pragma omp parallel for schedule(static, 64) reduction(+:total_loss)
    for (int i = 0; i < M; i++) {
        const float* li = logits + (long)i * V;
        float* gi = grad + (long)i * V;
        int t = targets[i];

        float32x4_t vmax = vdupq_n_f32(-1e30f);
        int j = 0;
        for (; j < V4; j += 4) vmax = vmaxq_f32(vmax, vld1q_f32(li + j));
        float mx = vmaxvq_f32(vmax);
        for (; j < V; j++) if (li[j] > mx) mx = li[j];

        float32x4_t vmx = vdupq_n_f32(mx);
        float32x4_t vsum = vdupq_n_f32(0.0f);
        j = 0;
        for (; j < V4; j += 4) {
            float32x4_t e = fast_exp_neon(vsubq_f32(vld1q_f32(li + j), vmx));
            vst1q_f32(gi + j, e);
            vsum = vaddq_f32(vsum, e);
        }
        float sum = vaddvq_f32(vsum);
        for (; j < V; j++) { float e = expf(li[j] - mx); gi[j] = e; sum += e; }

        float scale = 1.0f / (sum * M);
        float32x4_t vs = vdupq_n_f32(scale);
        j = 0;
        for (; j < V4; j += 4) vst1q_f32(gi + j, vmulq_f32(vld1q_f32(gi + j), vs));
        for (; j < V; j++) gi[j] *= scale;

        total_loss += -logf(gi[t] * M + 1e-9f);
        gi[t] -= 1.0f / M;
    }
    return (float)(total_loss / M);
}

/* ===== INT32 TO FLOAT32 ===== */
void int32_to_float32(const int32_t* src, float* dst, int n, float scale) {
    #pragma omp parallel for schedule(static, 4096)
    for (int i = 0; i < n; i++) dst[i] = (float)src[i] * scale;
}

/* ===== ELEMENT-WISE OPS ===== */
void add_f32(const float* a, const float* b, float* out, int n) {
    #pragma omp parallel for schedule(static, 4096)
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}

void mul_f32(const float* a, const float* b, float* out, int n) {
    #pragma omp parallel for schedule(static, 4096)
    for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
}

/* ===== SGD WITH MOMENTUM ===== */
void sgd_momentum(float* param, float* grad, float* velocity,
                   int n, float lr, float momentum, float wd) {
    #pragma omp parallel for schedule(static, 4096)
    for (int i = 0; i < n; i++) {
        grad[i] += wd * param[i];
        velocity[i] = momentum * velocity[i] + grad[i];
        param[i] -= lr * velocity[i];
    }
}

/* ===== TILED FLOAT32 MATMUL with NEON 4x4 micro-kernel ===== */
/* Tile sizes tuned for Kunpeng 920: L1=64KB, L2=512KB, L3=24MB */
/* mc*kc*4 should fit in L2 (~512KB), kc*nc*4 should fit in L3 */
void matmul_f32(const float* A, const float* B, float* C, int M, int K, int N) {
    const int mc = 128;   /* rows of A tile */
    const int kc = 192;   /* depth tile (== K for D=192, so single pass) */
    const int nc = 256;   /* cols of B tile */

    memset(C, 0, (long)M * N * sizeof(float));

    #pragma omp parallel for schedule(dynamic, 1) collapse(2)
    for (int i0 = 0; i0 < M; i0 += mc) {
        for (int j0 = 0; j0 < N; j0 += nc) {
            int iend = (i0 + mc < M) ? i0 + mc : M;
            int jend = (j0 + nc < N) ? j0 + nc : N;

            for (int k0 = 0; k0 < K; k0 += kc) {
                int kend = (k0 + kc < K) ? k0 + kc : K;

                for (int i = i0; i < iend; i += 4) {
                    int ilim = (i + 4 < iend) ? i + 4 : iend;
                    for (int j = j0; j < jend; j += 4) {
                        int jlim = (j + 4 < jend) ? j + 4 : jend;

                        /* 4x4 micro-kernel with NEON */
                        if (ilim - i == 4 && jlim - j == 4) {
                            float32x4_t c0 = vld1q_f32(C + (long)i * N + j);
                            float32x4_t c1 = vld1q_f32(C + (long)(i+1) * N + j);
                            float32x4_t c2 = vld1q_f32(C + (long)(i+2) * N + j);
                            float32x4_t c3 = vld1q_f32(C + (long)(i+3) * N + j);

                            for (int k = k0; k < kend; k++) {
                                float32x4_t bk = vld1q_f32(B + (long)k * N + j);
                                c0 = vmlaq_n_f32(c0, bk, A[(long)i * K + k]);
                                c1 = vmlaq_n_f32(c1, bk, A[(long)(i+1) * K + k]);
                                c2 = vmlaq_n_f32(c2, bk, A[(long)(i+2) * K + k]);
                                c3 = vmlaq_n_f32(c3, bk, A[(long)(i+3) * K + k]);
                            }

                            vst1q_f32(C + (long)i * N + j, c0);
                            vst1q_f32(C + (long)(i+1) * N + j, c1);
                            vst1q_f32(C + (long)(i+2) * N + j, c2);
                            vst1q_f32(C + (long)(i+3) * N + j, c3);
                        } else {
                            /* Scalar fallback for edges */
                            for (int ii = i; ii < ilim; ii++) {
                                for (int k = k0; k < kend; k++) {
                                    float a = A[(long)ii * K + k];
                                    for (int jj = j; jj < jlim; jj++) {
                                        C[(long)ii * N + jj] += a * B[(long)k * N + jj];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/* ===== MATMUL A.T @ B: C[K,N] = A[M,K].T @ B[M,N] ===== */
void matmul_atb_f32(const float* A, const float* B, float* C, int M, int K, int N) {
    long KN = (long)K * N;
    int nthreads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
    }

    float** local_C = (float**)malloc(nthreads * sizeof(float*));
    for (int t = 0; t < nthreads; t++) {
        local_C[t] = (float*)calloc(KN, sizeof(float));
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float* my_C = local_C[tid];

        #pragma omp for schedule(static, 256)
        for (int i = 0; i < M; i++) {
            const float* ai = A + (long)i * K;
            const float* bi = B + (long)i * N;
            for (int k = 0; k < K; k++) {
                float a = ai[k];
                float* ck = my_C + (long)k * N;
                for (int j = 0; j < N; j++) {
                    ck[j] += a * bi[j];
                }
            }
        }
    }

    #pragma omp parallel for schedule(static, 1024)
    for (long idx = 0; idx < KN; idx++) {
        float sum = 0.0f;
        for (int t = 0; t < nthreads; t++) sum += local_C[t][idx];
        C[idx] = sum;
    }

    for (int t = 0; t < nthreads; t++) free(local_C[t]);
    free(local_C);
}

/* ===== FUSED CE + EMBED BACKWARD ===== */
/* Computes loss, dx = ce_softmax_grad @ embed, d_embed += ce_softmax_grad.T @ x_final */
/* Processes row-by-row so we never store full (M,V) gradient */
void cross_entropy_bwd_fused(
    const float* logits,    /* M x V */
    const int32_t* targets, /* M */
    const float* x_final,   /* M x D */
    const float* embed,     /* V x D */
    float* dx,              /* M x D output */
    float* d_embed,         /* V x D output (accumulated) */
    float* loss_out,        /* scalar output */
    int M, int V, int D
) {
    memset(dx, 0, (long)M * D * sizeof(float));
    double total_loss = 0.0;

    int nthreads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
    }

    float** local_de = (float**)malloc(nthreads * sizeof(float*));
    for (int t = 0; t < nthreads; t++) {
        local_de[t] = (float*)calloc((long)V * D, sizeof(float));
    }

    #pragma omp parallel reduction(+:total_loss)
    {
        int tid = omp_get_thread_num();
        float* my_de = local_de[tid];
        float* probs = (float*)malloc(V * sizeof(float));

        #pragma omp for schedule(static, 64)
        for (int i = 0; i < M; i++) {
            const float* li = logits + (long)i * V;
            int tgt = targets[i];

            /* Softmax */
            float mx = li[0];
            for (int v = 1; v < V; v++) if (li[v] > mx) mx = li[v];
            float sum = 0.0f;
            for (int v = 0; v < V; v++) { probs[v] = expf(li[v] - mx); sum += probs[v]; }
            float inv_sum = 1.0f / sum;
            for (int v = 0; v < V; v++) probs[v] *= inv_sum;

            total_loss += -logf(probs[tgt] + 1e-9f);

            /* Gradient: g[v] = (probs[v] - onehot[v]) / M */
            probs[tgt] -= 1.0f;
            float inv_M = 1.0f / M;

            /* dx[i] = sum_v g[v] * embed[v], d_embed[v] += g[v] * x[i] */
            const float* xi = x_final + (long)i * D;
            float* dxi = dx + (long)i * D;

            for (int v = 0; v < V; v++) {
                float g = probs[v] * inv_M;
                if (fabsf(g) < 1e-7f) continue; /* skip near-zero grads */
                const float* ev = embed + (long)v * D;
                float* dev = my_de + (long)v * D;
                for (int d = 0; d < D; d++) {
                    dxi[d] += g * ev[d];
                    dev[d] += g * xi[d];
                }
            }
        }
        free(probs);
    }

    /* Reduce d_embed */
    #pragma omp parallel for schedule(static, 1024)
    for (long idx = 0; idx < (long)V * D; idx++) {
        float sum = 0.0f;
        for (int t = 0; t < nthreads; t++) sum += local_de[t][idx];
        d_embed[idx] += sum;
    }

    for (int t = 0; t < nthreads; t++) free(local_de[t]);
    free(local_de);

    *loss_out = (float)(total_loss / M);
}

/* ===== EMBEDDING LOOKUP ===== */
void embed_lookup(const float* embed, const int32_t* ids, float* out, int M, int D) {
    #pragma omp parallel for schedule(static, 256)
    for (int i = 0; i < M; i++) {
        int id = ids[i];
        const float* src = embed + (long)id * D;
        float* dst = out + (long)i * D;
        memcpy(dst, src, D * sizeof(float));
    }
}

/* ===== EMBEDDING GRADIENT SCATTER (parallel with thread-local reduce) ===== */
/* d_embed[ids[i]] += dx[i] for each i */
void embed_grad_scatter(float* d_embed, const int32_t* ids, const float* dx, int M, int D) {
    int nthreads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
    }

    /* With V=1024, D=192 → 768 KB per copy, manageable */
    const int V = 1024;
    float** local_de = (float**)malloc(nthreads * sizeof(float*));
    for (int t = 0; t < nthreads; t++) {
        local_de[t] = (float*)calloc((long)V * D, sizeof(float));
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float* my_de = local_de[tid];
        #pragma omp for schedule(static, 256)
        for (int i = 0; i < M; i++) {
            int id = ids[i];
            const float* src = dx + (long)i * D;
            float* dst = my_de + (long)id * D;
            for (int d = 0; d < D; d++) dst[d] += src[d];
        }
    }

    /* Reduce to global d_embed */
    long VD = (long)V * D;
    #pragma omp parallel for schedule(static, 1024)
    for (long idx = 0; idx < VD; idx++) {
        float sum = 0.0f;
        for (int t = 0; t < nthreads; t++) sum += local_de[t][idx];
        d_embed[idx] += sum;
    }

    for (int t = 0; t < nthreads; t++) free(local_de[t]);
    free(local_de);
}

/* ===== WEIGHT QUANTIZE: float32 shadow -> packed ternary ===== */
/* BitNet b1.58: round(clip(W / mean(|W|), -1, 1)) */
void quantize_weights(const float* W, uint8_t* val, uint8_t* sign,
                       float* scale_out, int rows, int cols) {
    int KB = (cols + 7) / 8;
    long total = (long)rows * cols;

    /* Compute mean(|W|) */
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static, 4096)
    for (long i = 0; i < total; i++) sum += fabsf(W[i]);
    float scale = (float)(sum / total) + 1e-8f;
    *scale_out = scale;

    /* Quantize and pack */
    #pragma omp parallel for schedule(static, 32)
    for (int r = 0; r < rows; r++) {
        const float* wr = W + (long)r * cols;
        uint8_t* vr = val + (long)r * KB;
        uint8_t* sr = sign + (long)r * KB;
        memset(vr, 0, KB);
        memset(sr, 0, KB);
        for (int c = 0; c < cols; c++) {
            float q = wr[c] / scale;
            if (q > 0.5f) {
                int bi = c / 8, bt = 7 - (c % 8);
                vr[bi] |= (1 << bt);
            } else if (q < -0.5f) {
                int bi = c / 8, bt = 7 - (c % 8);
                vr[bi] |= (1 << bt);
                sr[bi] |= (1 << bt);
            }
        }
    }
}

/* ===== UNPACK TERNARY -> FLOAT32 ===== */
void unpack_ternary_f32(const uint8_t* val, const uint8_t* sign,
                         float* out, int rows, int cols) {
    int KB = (cols + 7) / 8;
    #pragma omp parallel for schedule(static, 64)
    for (int i = 0; i < rows; i++) {
        const uint8_t* vi = val + (long)i * KB;
        const uint8_t* si = sign + (long)i * KB;
        float* oi = out + (long)i * cols;
        for (int j = 0; j < cols; j++) {
            int bi = j / 8, bt = 7 - (j % 8);
            uint8_t mask = 1 << bt;
            if (vi[bi] & mask) {
                oi[j] = (si[bi] & mask) ? -1.0f : 1.0f;
            } else {
                oi[j] = 0.0f;
            }
        }
    }
}