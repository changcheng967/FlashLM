/*
 * ternary_engine.c — FlashLM NEON/AVX2 Cross-Platform Training Engine
 * 
 * ARM aarch64: uses NEON intrinsics (vandq_u8, vcntq_u8, etc.)
 * x86_64:      uses AVX2 + POPCNT (or vpshufb nibble lookup)
 * 
 * Compile:
 *   ARM:  gcc -O3 -march=armv8-a+simd -fopenmp -shared -fPIC -lm -o ternary_engine.so ternary_engine.c
 *   x86:  gcc -O3 -march=native -mavx2 -mpopcnt -fopenmp -shared -fPIC -lm -o ternary_engine.so ternary_engine.c
 */

#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

#ifdef __aarch64__
  #include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
  #include <x86intrin.h>
#endif

/* ================================================================
 * HELPER: POPCOUNT for x86 AVX2
 * Nibble-lookup method — works on all AVX2 CPUs (Haswell+, Zen 1+)
 * ================================================================ */
#if defined(__x86_64__) || defined(_M_X64)

static inline __m256i avx2_popcount_u8(__m256i v) {
    /* Nibble lookup table: popcount of 0..15 */
    const __m256i lookup = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4
    );
    __m256i low_mask = _mm256_set1_epi8(0x0F);
    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    return _mm256_add_epi8(
        _mm256_shuffle_epi8(lookup, lo),
        _mm256_shuffle_epi8(lookup, hi)
    );
}

/* Sum all bytes in a __m256i, return as int32 */
static inline int32_t avx2_sum_u8(__m256i v) {
    /* sad against zero gives 4x uint64 partial sums */
    __m256i sad = _mm256_sad_epu8(v, _mm256_setzero_si256());
    /* Extract and sum the 4 x 64-bit values */
    __m128i lo = _mm256_castsi256_si128(sad);
    __m128i hi = _mm256_extracti128_si256(sad, 1);
    __m128i sum = _mm_add_epi64(lo, hi);
    return (int32_t)(_mm_extract_epi64(sum, 0) + _mm_extract_epi64(sum, 1));
}

/* Fast exp for x86 AVX2 — Schraudolph + polynomial */
static inline __m256 fast_exp_avx2(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));
    
    __m256 log2e = _mm256_set1_ps(1.44269504f);
    __m256 t = _mm256_mul_ps(x, log2e);
    
    /* n = floor(t) */
    __m256 nf = _mm256_floor_ps(t);
    __m256i n = _mm256_cvtps_epi32(nf);
    
    /* f = t - n */
    __m256 f = _mm256_sub_ps(t, nf);
    
    /* 2^f polynomial: 1 + f*ln2 + f^2*ln2^2/2 + f^3*ln2^3/6 */
    __m256 c1 = _mm256_set1_ps(0.693147180f);
    __m256 c2 = _mm256_set1_ps(0.240226507f);
    __m256 c3 = _mm256_set1_ps(0.055504109f);
    __m256 p = _mm256_fmadd_ps(c3, f, c2);
    p = _mm256_fmadd_ps(p, f, c1);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));
    
    /* Multiply by 2^n */
    __m256i exp_bits = _mm256_slli_epi32(n, 23);
    __m256 pow2n = _mm256_castsi256_ps(
        _mm256_add_epi32(_mm256_castps_si256(_mm256_set1_ps(1.0f)), exp_bits)
    );
    
    return _mm256_mul_ps(p, pow2n);
}

#endif /* x86_64 */


/* ================================================================
 * FAST EXP for ARM NEON
 * ================================================================ */
#ifdef __aarch64__

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
        vaddq_s32(vreinterpretq_s32_f32(vdupq_n_f32(1.0f)), exp_bits)
    );
    
    return vmulq_f32(p, pow2n);
}

#endif /* aarch64 */


/* ================================================================
 * TERNARY MATMUL: C = X_ternary @ W_ternary.T
 * X: M x KB (packed), W: N x KB (packed), C: M x N (int32)
 * ================================================================ */
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

#ifdef __aarch64__
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
#elif defined(__x86_64__)
        for (; k + 32 <= KB; k += 32) {
            __m256i xv = _mm256_loadu_si256((__m256i*)(xi_v + k));
            __m256i xs = _mm256_loadu_si256((__m256i*)(xi_s + k));
            __m256i wv = _mm256_loadu_si256((__m256i*)(wj_v + k));
            __m256i ws = _mm256_loadu_si256((__m256i*)(wj_s + k));
            __m256i bv = _mm256_and_si256(xv, wv);
            __m256i bs = _mm256_and_si256(_mm256_xor_si256(xs, ws), bv);
            ps += avx2_sum_u8(avx2_popcount_u8(bv));
            ns += avx2_sum_u8(avx2_popcount_u8(bs));
        }
#endif
        /* Scalar fallback for remaining bytes */
        for (; k < KB; k++) {
            uint8_t bv = xi_v[k] & wj_v[k];
            uint8_t bs = (xi_s[k] ^ wj_s[k]) & bv;
            ps += __builtin_popcount(bv);
            ns += __builtin_popcount(bs);
        }
        C[idx] = ps - 2 * ns;
    }
}


/* ================================================================
 * TERNARY WEIGHT GRADIENT: dW = X_ternary.T @ grad_float
 * Scatter-add with per-thread local buffers
 * ================================================================ */
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
                uint8_t vb = vi[byte_idx];

                if (!(vb & mask)) continue;

                float* dw_row = my_dW + (long)k * N;
                if (si[byte_idx] & mask) {
                    int n = 0;
                    for (; n + 3 < N; n += 4) {
                        dw_row[n]   -= gi[n];
                        dw_row[n+1] -= gi[n+1];
                        dw_row[n+2] -= gi[n+2];
                        dw_row[n+3] -= gi[n+3];
                    }
                    for (; n < N; n++) dw_row[n] -= gi[n];
                } else {
                    int n = 0;
                    for (; n + 3 < N; n += 4) {
                        dw_row[n]   += gi[n];
                        dw_row[n+1] += gi[n+1];
                        dw_row[n+2] += gi[n+2];
                        dw_row[n+3] += gi[n+3];
                    }
                    for (; n < N; n++) dw_row[n] += gi[n];
                }
            }
        }
    }

    #pragma omp parallel for schedule(static, 1024)
    for (long idx = 0; idx < KN; idx++) {
        float sum = 0.0f;
        for (int t = 0; t < nthreads; t++) {
            sum += local_dW[t][idx];
        }
        dW[idx] = sum;
    }

    for (int t = 0; t < nthreads; t++) free(local_dW[t]);
    free(local_dW);
}


/* ================================================================
 * SiLU IN-PLACE
 * ================================================================ */
void silu_f32(float* x, int n) {
#ifdef __aarch64__
    #pragma omp parallel for schedule(static, 4096)
    for (int i = 0; i < n - 3; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t neg_v = vnegq_f32(v);
        float32x4_t sig = vrecpeq_f32(vaddq_f32(vdupq_n_f32(1.0f), fast_exp_neon(neg_v)));
        sig = vmulq_f32(sig, vrecpsq_f32(sig, vaddq_f32(vdupq_n_f32(1.0f), fast_exp_neon(neg_v))));
        vst1q_f32(x + i, vmulq_f32(v, sig));
    }
    for (int i = (n & ~3); i < n; i++) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
#elif defined(__x86_64__)
    #pragma omp parallel for schedule(static, 4096)
    for (int i = 0; i < n - 7; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 neg_v = _mm256_sub_ps(_mm256_setzero_ps(), v);
        __m256 e = fast_exp_avx2(neg_v);
        __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f), e);
        __m256 sig = _mm256_div_ps(_mm256_set1_ps(1.0f), denom);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(v, sig));
    }
    for (int i = (n & ~7); i < n; i++) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
#else
    #pragma omp parallel for schedule(static, 4096)
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
#endif
}


/* ================================================================
 * SiLU BACKWARD
 * ================================================================ */
void silu_bwd_f32(const float* x, const float* grad_out, float* grad_in, int n) {
#ifdef __aarch64__
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
#elif defined(__x86_64__)
    __m256 one = _mm256_set1_ps(1.0f);
    #pragma omp parallel for schedule(static, 4096)
    for (int i = 0; i < n - 7; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 go = _mm256_loadu_ps(grad_out + i);
        __m256 neg_v = _mm256_sub_ps(_mm256_setzero_ps(), v);
        __m256 e = fast_exp_avx2(neg_v);
        __m256 denom = _mm256_add_ps(one, e);
        __m256 sig = _mm256_div_ps(one, denom);
        __m256 one_minus_sig = _mm256_sub_ps(one, sig);
        /* dsilu = sig * (1 + x * (1 - sig)) */
        __m256 dsilu = _mm256_mul_ps(sig, _mm256_fmadd_ps(v, one_minus_sig, one));
        _mm256_storeu_ps(grad_in + i, _mm256_mul_ps(go, dsilu));
    }
    for (int i = (n & ~7); i < n; i++) {
        float s = 1.0f / (1.0f + expf(-x[i]));
        grad_in[i] = grad_out[i] * (s + x[i] * s * (1.0f - s));
    }
#else
    #pragma omp parallel for schedule(static, 4096)
    for (int i = 0; i < n; i++) {
        float s = 1.0f / (1.0f + expf(-x[i]));
        grad_in[i] = grad_out[i] * (s + x[i] * s * (1.0f - s));
    }
#endif
}


/* ================================================================
 * RMSNORM FORWARD
 * ================================================================ */
void rmsnorm_f32(const float* x, const float* gamma, float* out, int M, int D) {
    #pragma omp parallel for schedule(static, 32)
    for (int i = 0; i < M; i++) {
        const float* xi = x + (long)i * D;
        float* oi = out + (long)i * D;
        float sum = 0.0f;
        for (int j = 0; j < D; j++) sum += xi[j] * xi[j];
        float scale = 1.0f / sqrtf(sum / D + 1e-6f);
        for (int j = 0; j < D; j++) oi[j] = xi[j] * scale * gamma[j];
    }
}


/* ================================================================
 * RMSNORM BACKWARD
 * ================================================================ */
void rmsnorm_bwd_f32(const float* x, const float* gamma, const float* grad_out,
                      float* grad_gamma, float* grad_in, int M, int D) {
    memset(grad_gamma, 0, D * sizeof(float));

    #pragma omp parallel
    {
        float* local_gg = (float*)calloc(D, sizeof(float));
        #pragma omp for schedule(static, 32)
        for (int i = 0; i < M; i++) {
            const float* xi = x + (long)i * D;
            const float* go = grad_out + (long)i * D;
            float* gi = grad_in + (long)i * D;
            float sum = 0.0f;
            for (int j = 0; j < D; j++) sum += xi[j] * xi[j];
            float rms2 = sum / D + 1e-6f;
            float inv_rms = 1.0f / sqrtf(rms2);
            float dot = 0.0f;
            for (int j = 0; j < D; j++) dot += go[j] * gamma[j] * xi[j];
            dot *= inv_rms * inv_rms / D;
            for (int j = 0; j < D; j++) {
                gi[j] = (go[j] * gamma[j] - xi[j] * dot) * inv_rms;
                local_gg[j] += go[j] * xi[j] * inv_rms;
            }
        }
        #pragma omp critical
        for (int j = 0; j < D; j++) grad_gamma[j] += local_gg[j];
        free(local_gg);
    }
}


/* ================================================================
 * REQUANTIZE: float32 -> packed ternary
 * ================================================================ */
void requantize_f32(const float* x, uint8_t* out_val, uint8_t* out_sign, int M, int D) {
    int KB = (D + 7) / 8;
    #pragma omp parallel for schedule(static, 32)
    for (int i = 0; i < M; i++) {
        const float* xi = x + (long)i * D;
        uint8_t* ov = out_val + (long)i * KB;
        uint8_t* os = out_sign + (long)i * KB;
        float sum = 0.0f;
        for (int j = 0; j < D; j++) sum += fabsf(xi[j]);
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


/* ================================================================
 * CROSS-ENTROPY FORWARD + BACKWARD (FUSED)
 * ================================================================ */
float cross_entropy_fwd_bwd(const float* logits, const int32_t* targets,
                             float* grad, int M, int V) {
    double total_loss = 0.0;

#ifdef __aarch64__
    int V4 = V & ~3;
    #pragma omp parallel for schedule(static, 64) reduction(+:total_loss)
    for (int i = 0; i < M; i++) {
        const float* li = logits + (long)i * V;
        float* gi = grad + (long)i * V;
        int t = targets[i];
        
        float32x4_t vmax = vdupq_n_f32(-1e30f);
        int j = 0;
        for (; j < V4; j += 4) {
            vmax = vmaxq_f32(vmax, vld1q_f32(li + j));
        }
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

#elif defined(__x86_64__)
    int V8 = V & ~7;
    #pragma omp parallel for schedule(static, 64) reduction(+:total_loss)
    for (int i = 0; i < M; i++) {
        const float* li = logits + (long)i * V;
        float* gi = grad + (long)i * V;
        int t = targets[i];
        
        /* Find max */
        __m256 vmax = _mm256_set1_ps(-1e30f);
        int j = 0;
        for (; j < V8; j += 8) {
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(li + j));
        }
        /* Horizontal max of 8 floats */
        __m128 hi128 = _mm256_extractf128_ps(vmax, 1);
        __m128 lo128 = _mm256_castps256_ps128(vmax);
        __m128 m128 = _mm_max_ps(lo128, hi128);
        m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
        m128 = _mm_max_ss(m128, _mm_movehdup_ps(m128));
        float mx = _mm_cvtss_f32(m128);
        for (; j < V; j++) if (li[j] > mx) mx = li[j];
        
        /* exp(x - max) and sum */
        __m256 vmx = _mm256_set1_ps(mx);
        __m256 vsum = _mm256_setzero_ps();
        j = 0;
        for (; j < V8; j += 8) {
            __m256 e = fast_exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(li + j), vmx));
            _mm256_storeu_ps(gi + j, e);
            vsum = _mm256_add_ps(vsum, e);
        }
        /* Horizontal sum */
        __m128 s128 = _mm_add_ps(_mm256_castps256_ps128(vsum), _mm256_extractf128_ps(vsum, 1));
        s128 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
        s128 = _mm_add_ss(s128, _mm_movehdup_ps(s128));
        float sum = _mm_cvtss_f32(s128);
        for (; j < V; j++) { float e = expf(li[j] - mx); gi[j] = e; sum += e; }
        
        /* Normalize */
        float scale = 1.0f / (sum * M);
        __m256 vs = _mm256_set1_ps(scale);
        j = 0;
        for (; j < V8; j += 8) _mm256_storeu_ps(gi + j, _mm256_mul_ps(_mm256_loadu_ps(gi + j), vs));
        for (; j < V; j++) gi[j] *= scale;
        
        total_loss += -logf(gi[t] * M + 1e-9f);
        gi[t] -= 1.0f / M;
    }

#else
    /* Pure scalar fallback */
    #pragma omp parallel for schedule(static, 64) reduction(+:total_loss)
    for (int i = 0; i < M; i++) {
        const float* li = logits + (long)i * V;
        float* gi = grad + (long)i * V;
        int t = targets[i];
        float mx = li[0];
        for (int j = 1; j < V; j++) if (li[j] > mx) mx = li[j];
        float sum = 0.0f;
        for (int j = 0; j < V; j++) { float e = expf(li[j] - mx); gi[j] = e; sum += e; }
        float scale = 1.0f / (sum * M);
        for (int j = 0; j < V; j++) gi[j] *= scale;
        total_loss += -logf(gi[t] * M + 1e-9f);
        gi[t] -= 1.0f / M;
    }
#endif

    return (float)(total_loss / M);
}


/* ================================================================
 * INT32 TO FLOAT32
 * ================================================================ */
void int32_to_float32(const int32_t* src, float* dst, int n) {
    #pragma omp parallel for schedule(static, 4096)
    for (int i = 0; i < n; i++) {
        dst[i] = (float)src[i];
    }
}


/* ================================================================
 * ELEMENT-WISE OPS
 * ================================================================ */
void add_f32(const float* a, const float* b, float* out, int n) {
    #pragma omp parallel for schedule(static, 4096)
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}

void mul_f32(const float* a, const float* b, float* out, int n) {
    #pragma omp parallel for schedule(static, 4096)
    for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
}


/* ================================================================
 * SGD WITH MOMENTUM
 * ================================================================ */
void sgd_momentum(float* param, float* grad, float* velocity,
                   int n, float lr, float momentum, float wd) {
    #pragma omp parallel for schedule(static, 4096)
    for (int i = 0; i < n; i++) {
        grad[i] += wd * param[i];
        velocity[i] = momentum * velocity[i] + grad[i];
        param[i] -= lr * velocity[i];
    }
}
