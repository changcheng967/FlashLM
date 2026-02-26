#include <arm_neon.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ===== TERNARY MATMUL ===== */
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

/* ===== TERNARY-FLOAT WEIGHT GRADIENT: dW = X_ternary.T @ grad_float ===== */
void ternary_transpose_matmul_f32(
    const uint8_t* X_val, const uint8_t* X_sign,
    const float* grad, float* dW,
    int M, int K, int N
) {
    int KB = (K + 7) / 8;
    memset(dW, 0, (long)K * N * sizeof(float));

    #pragma omp parallel for schedule(static, 4) collapse(2)
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            int byte_idx = k / 8;
            int bit_idx = 7 - (k % 8);
            uint8_t mask = 1 << bit_idx;
            float sum = 0.0f;
            int i = 0;
            for (; i + 3 < M; i += 4) {
                uint8_t v0 = X_val[(long)i * KB + byte_idx];
                uint8_t s0 = X_sign[(long)i * KB + byte_idx];
                uint8_t v1 = X_val[(long)(i+1) * KB + byte_idx];
                uint8_t s1 = X_sign[(long)(i+1) * KB + byte_idx];
                uint8_t v2 = X_val[(long)(i+2) * KB + byte_idx];
                uint8_t s2 = X_sign[(long)(i+2) * KB + byte_idx];
                uint8_t v3 = X_val[(long)(i+3) * KB + byte_idx];
                uint8_t s3 = X_sign[(long)(i+3) * KB + byte_idx];
                if (v0 & mask) sum += (s0 & mask) ? -grad[(long)i * N + n] : grad[(long)i * N + n];
                if (v1 & mask) sum += (s1 & mask) ? -grad[(long)(i+1) * N + n] : grad[(long)(i+1) * N + n];
                if (v2 & mask) sum += (s2 & mask) ? -grad[(long)(i+2) * N + n] : grad[(long)(i+2) * N + n];
                if (v3 & mask) sum += (s3 & mask) ? -grad[(long)(i+3) * N + n] : grad[(long)(i+3) * N + n];
            }
            for (; i < M; i++) {
                uint8_t v = X_val[(long)i * KB + byte_idx];
                uint8_t s = X_sign[(long)i * KB + byte_idx];
                if (v & mask) sum += (s & mask) ? -grad[(long)i * N + n] : grad[(long)i * N + n];
            }
            dW[(long)k * N + n] = sum;
        }
    }
}

/* ===== SiLU IN-PLACE ===== */
void silu_f32(float* x, int n) {
    #pragma omp parallel for schedule(static, 4096)
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

/* ===== SiLU BACKWARD ===== */
void silu_bwd_f32(const float* x, const float* grad_out, float* grad_in, int n) {
    #pragma omp parallel for schedule(static, 4096)
    for (int i = 0; i < n; i++) {
        float s = 1.0f / (1.0f + expf(-x[i]));
        grad_in[i] = grad_out[i] * (s + x[i] * s * (1.0f - s));
    }
}

/* ===== RMSNORM ===== */
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

/* ===== RMSNORM BACKWARD ===== */
void rmsnorm_bwd_f32(const float* x, const float* gamma, const float* grad_out,
                      float* grad_in, float* grad_gamma, int M, int D) {
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

/* ===== REQUANTIZE: float32 -> packed ternary ===== */
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

/* ===== CROSS-ENTROPY FORWARD + BACKWARD (FUSED) ===== */
float cross_entropy_fwd_bwd(const float* logits, const int32_t* targets,
                             float* grad, int M, int V) {
    double total_loss = 0.0;
    #pragma omp parallel for schedule(static, 32) reduction(+:total_loss)
    for (int i = 0; i < M; i++) {
        const float* li = logits + (long)i * V;
        float* gi = grad + (long)i * V;
        int t = targets[i];
        float mx = -1e30f;
        for (int j = 0; j < V; j++) if (li[j] > mx) mx = li[j];
        float sum = 0.0f;
        for (int j = 0; j < V; j++) {
            gi[j] = expf(li[j] - mx);
            sum += gi[j];
        }
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < V; j++) gi[j] *= inv_sum;
        total_loss += -logf(gi[t] + 1e-9f);
        gi[t] -= 1.0f;
        float scale = 1.0f / M;
        for (int j = 0; j < V; j++) gi[j] *= scale;
    }
    return (float)(total_loss / M);
}

/* ===== INT32 TO FLOAT32 WITH SCALE ===== */
void int32_to_float32(const int32_t* src, float* dst, int n, float scale) {
    #pragma omp parallel for schedule(static, 4096)
    for (int i = 0; i < n; i++) {
        dst[i] = (float)src[i] * scale;
    }
}

/* ===== ELEMENT-WISE ADD ===== */
void add_f32(const float* a, const float* b, float* out, int n) {
    #pragma omp parallel for schedule(static, 4096)
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}

/* ===== ELEMENT-WISE MULTIPLY ===== */
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
