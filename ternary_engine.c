#include <arm_neon.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

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

/* ===== SiLU IN-PLACE ===== */
void silu_f32(float* x, int n) {
    #pragma omp parallel for schedule(static, 4096)
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

/* ===== SiLU BACKWARD: grad_in = grad_out * (sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))) ===== */
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
    // Zero grad_gamma
    memset(grad_gamma, 0, D * sizeof(float));
    float* tmp_gg = (float*)calloc(D, sizeof(float));

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
    free(tmp_gg);
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
        // Find max for numerical stability
        float mx = -1e30f;
        for (int j = 0; j < V; j++) if (li[j] > mx) mx = li[j];
        // Compute softmax and gradient
        float sum = 0.0f;
        for (int j = 0; j < V; j++) {
            gi[j] = expf(li[j] - mx);
            sum += gi[j];
        }
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < V; j++) gi[j] *= inv_sum;
        total_loss += -logf(gi[t] + 1e-9f);
        gi[t] -= 1.0f;
        // Scale gradient by 1/M
        float scale = 1.0f / M;
        for (int j = 0; j < V; j++) gi[j] *= scale;
    }
    return (float)(total_loss / M);
}

/* ===== MATMUL FLOAT32 (for embedding/lm_head grad) ===== */
void matmul_f32(const float* A, const float* B, float* C,
                int M, int K, int N) {
    // C = A @ B, A is MxK, B is KxN, C is MxN
    memset(C, 0, (long)M * N * sizeof(float));
    #pragma omp parallel for schedule(static, 16)
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a = A[(long)i * K + k];
            for (int j = 0; j < N; j++) {
                C[(long)i * N + j] += a * B[(long)k * N + j];
            }
        }
    }
}

/* ===== MATMUL_AT_B: C = A^T @ B (for weight gradients) ===== */
void matmul_atb_f32(const float* A, const float* B, float* C,
                     int M, int K, int N) {
    // A is MxK, B is MxN, C = A^T @ B = KxN
    memset(C, 0, (long)K * N * sizeof(float));
    #pragma omp parallel for schedule(static, 4)
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < M; i++) {
            float a = A[(long)i * K + k];
            for (int j = 0; j < N; j++) {
                C[(long)k * N + j] += a * B[(long)i * N + j];
            }
        }
    }
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
