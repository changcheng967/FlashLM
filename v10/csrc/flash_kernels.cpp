/*
 * flash_kernels.cpp — CPU-optimized kernels for FLASH architecture
 *
 * Fused operations that replace ~15 PyTorch dispatch calls with 1 C++ call:
 *   1. GLT-Linear: quantize + index + table lookup (fused forward+backward)
 *   2. SSM Scan: sequential EMA (fused forward+backward)
 *   3. Walsh-Hadamard: O(d log d) instead of O(d^2) matmul
 *
 * Compile flags: -O3 -mavx2 -fopenmp
 */

#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <algorithm>

// ============================================================================
// GLT-Linear Forward: fused quantize + index + table lookup
// ============================================================================
// Replaces: tanh, scale, round, clamp, view, 4x index loop, embedding, reshape
// Output: (output tensor, saved indices for backward)

std::vector<torch::Tensor> glt_forward(
    const torch::Tensor& x,       // (N, d_in) float32
    const torch::Tensor& tables,  // (num_groups, entries, d_out_per_group) float32
    int64_t bits,
    int64_t dims_per_group)
{
    const int N = x.size(0);
    const int d_in = x.size(1);
    const int num_groups = d_in / (int)dims_per_group;
    const int n_levels = 1 << bits;
    const int entries = tables.size(1);
    const int dog = tables.size(2);  // d_out_per_group
    const int d_out = num_groups * dog;

    auto output = torch::zeros({N, d_out}, x.options());
    auto indices = torch::empty({N, num_groups}, x.options().dtype(torch::kLong));

    const float* __restrict__ x_ptr = x.data_ptr<float>();
    const float* __restrict__ t_ptr = tables.data_ptr<float>();
    float* __restrict__ o_ptr = output.data_ptr<float>();
    int64_t* __restrict__ i_ptr = indices.data_ptr<int64_t>();

    const float scale = (float)(n_levels - 1) * 0.5f;

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        for (int g = 0; g < num_groups; g++) {
            // Quantize + compute mixed-radix index
            int idx = 0;
            int mult = 1;
            for (int d = 0; d < (int)dims_per_group; d++) {
                float val = x_ptr[n * d_in + g * (int)dims_per_group + d];
                val = tanhf(val);
                val = (val + 1.0f) * scale;
                int q = (int)lroundf(val);
                if (q < 0) q = 0;
                if (q >= n_levels) q = n_levels - 1;
                idx += q * mult;
                mult *= n_levels;
            }
            if (idx >= entries) idx = entries - 1;
            i_ptr[n * num_groups + g] = idx;

            // Table lookup: copy d_out_per_group floats
            const float* row = t_ptr + (g * entries + idx) * dog;
            float* out = o_ptr + n * d_out + g * dog;
            for (int j = 0; j < dog; j++) {
                out[j] = row[j];
            }
        }
    }

    return {output, indices};
}

// GLT-Linear Backward: scatter-add to tables, STE identity for input
std::vector<torch::Tensor> glt_backward(
    const torch::Tensor& grad_output,  // (N, d_out)
    const torch::Tensor& indices,      // (N, num_groups) saved from forward
    int64_t num_groups_in,
    int64_t entries_in,
    int64_t dog_in,
    int64_t d_in)
{
    const int N = grad_output.size(0);
    const int d_out = grad_output.size(1);
    const int num_groups = (int)num_groups_in;
    const int entries = (int)entries_in;
    const int dog = (int)dog_in;

    // Gradient to tables: scatter-add
    auto grad_tables = torch::zeros({num_groups, entries, dog}, grad_output.options());

    const float* __restrict__ go_ptr = grad_output.data_ptr<float>();
    const int64_t* __restrict__ i_ptr = indices.data_ptr<int64_t>();
    float* __restrict__ gt_ptr = grad_tables.data_ptr<float>();

    // Use thread-local accumulation to avoid atomic contention
    const int nthreads = omp_get_max_threads();
    auto local_grads = std::vector<torch::Tensor>(nthreads);
    for (int t = 0; t < nthreads; t++) {
        local_grads[t] = torch::zeros({num_groups, entries, dog}, grad_output.options());
    }

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        int tid = omp_get_thread_num();
        float* __restrict__ lg = local_grads[tid].data_ptr<float>();
        for (int g = 0; g < num_groups; g++) {
            int idx = (int)i_ptr[n * num_groups + g];
            const float* go_row = go_ptr + n * d_out + g * dog;
            float* lg_row = lg + (g * entries + idx) * dog;
            for (int j = 0; j < dog; j++) {
                lg_row[j] += go_row[j];
            }
        }
    }

    // Reduce thread-local gradients
    for (int t = 0; t < nthreads; t++) {
        grad_tables.add_(local_grads[t]);
    }

    // STE: gradient to input is zero (bypass handles gradient flow)
    auto grad_input = torch::zeros({N, (int)d_in}, grad_output.options());

    return {grad_input, grad_tables};
}

// ============================================================================
// SSM Scan Forward: h_t = decay * h_{t-1} + x_t
// ============================================================================

torch::Tensor ssm_scan_forward(
    const torch::Tensor& x,     // (B, T, d)
    const torch::Tensor& decay) // (d,)
{
    const int B = x.size(0);
    const int T = x.size(1);
    const int d = x.size(2);

    auto output = torch::empty_like(x);

    const float* __restrict__ x_ptr = x.data_ptr<float>();
    const float* __restrict__ d_ptr = decay.data_ptr<float>();
    float* __restrict__ o_ptr = output.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        const float* x_row = x_ptr + b * T * d;
        float* o_row = o_ptr + b * T * d;

        // t = 0: output = input
        for (int j = 0; j < d; j++) {
            o_row[j] = x_row[j];
        }

        // t = 1..T-1: recurrence
        for (int t = 1; t < T; t++) {
            const float* x_t = x_row + t * d;
            const float* o_prev = o_row + (t - 1) * d;
            float* o_t = o_row + t * d;
            for (int j = 0; j < d; j++) {
                o_t[j] = d_ptr[j] * o_prev[j] + x_t[j];
            }
        }
    }

    return output;
}

// SSM Scan Backward: reverse scan for grad_input + grad_decay
std::vector<torch::Tensor> ssm_scan_backward(
    const torch::Tensor& grad_output,  // (B, T, d)
    const torch::Tensor& output,       // (B, T, d) from forward (needed for decay grad)
    const torch::Tensor& x,            // (B, T, d) from forward
    const torch::Tensor& decay)        // (d,)
{
    const int B = grad_output.size(0);
    const int T = grad_output.size(1);
    const int d = grad_output.size(2);

    auto grad_input = torch::empty_like(grad_output);
    auto grad_decay = torch::zeros_like(decay);

    const float* __restrict__ go_ptr = grad_output.data_ptr<float>();
    const float* __restrict__ o_ptr = output.data_ptr<float>();
    const float* __restrict__ x_ptr = x.data_ptr<float>();
    const float* __restrict__ d_ptr = decay.data_ptr<float>();
    float* __restrict__ gi_ptr = grad_input.data_ptr<float>();

    // Thread-local decay gradients
    const int nthreads = omp_get_max_threads();
    std::vector<float*> gd_local(nthreads);
    for (int t = 0; t < nthreads; t++) {
        gd_local[t] = new float[d]();
    }

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        int tid = omp_get_thread_num();
        float* gd = gd_local[tid];

        const float* go_row = go_ptr + b * T * d;
        const float* o_row = o_ptr + b * T * d;
        float* gi_row = gi_ptr + b * T * d;

        // Reverse scan: accumulate gradient backward through time
        // g_t = grad_output_t + decay * g_{t+1}
        // grad_input_t = g_t
        std::vector<float> g_h(d, 0.0f);

        for (int t = T - 1; t >= 0; t--) {
            const float* go_t = go_row + t * d;
            float* gi_t = gi_row + t * d;

            for (int j = 0; j < d; j++) {
                g_h[j] = g_h[j] * d_ptr[j] + go_t[j];
                gi_t[j] = g_h[j];
            }

            // Accumulate decay gradient: d_loss/d_decay_j += h_{t-1,j} * g_h_j
            if (t > 0) {
                const float* o_prev = o_row + (t - 1) * d;
                for (int j = 0; j < d; j++) {
                    gd[j] += o_prev[j] * g_h[j];
                }
            }
        }
    }

    // Reduce decay gradients
    auto gd_acc = grad_decay.accessor<float, 1>();
    for (int t = 0; t < nthreads; t++) {
        for (int j = 0; j < d; j++) {
            gd_acc[j] += gd_local[t][j];
        }
        delete[] gd_local[t];
    }

    return {grad_input, grad_decay};
}

// ============================================================================
// Fast Walsh-Hadamard Transform: O(d log d) instead of O(d^2) matmul
// ============================================================================

torch::Tensor hadamard_forward(
    const torch::Tensor& x,     // (N, d) -- d must be power of 2
    const torch::Tensor& scale) // (d,) learnable per-dim scale
{
    const int N = x.size(0);
    const int d = x.size(1);

    auto output = x.clone();
    float* __restrict__ ptr = output.data_ptr<float>();
    const float* __restrict__ s_ptr = scale.data_ptr<float>();
    const float norm = 1.0f / sqrtf((float)d);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        float* row = ptr + n * d;

        // Butterfly stages: O(d log d) additions
        for (int h = 1; h < d; h *= 2) {
            for (int i = 0; i < d; i += 2 * h) {
                for (int j = 0; j < h; j++) {
                    float a = row[i + j];
                    float b = row[i + j + h];
                    row[i + j] = a + b;
                    row[i + j + h] = a - b;
                }
            }
        }

        // Normalize and apply learnable scale
        for (int j = 0; j < d; j++) {
            row[j] *= norm * s_ptr[j];
        }
    }

    return output;
}

// Hadamard backward: H is orthogonal (H * H^T = I), so backward = forward on grad
torch::Tensor hadamard_backward(
    const torch::Tensor& grad_output,  // (N, d)
    const torch::Tensor& scale)        // (d,)
{
    // grad_input = H(grad_output) * scale / sqrt(d)
    return hadamard_forward(grad_output, scale);
}

// ============================================================================
// Module
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("glt_forward", &glt_forward, "Fused GLT-Linear forward");
    m.def("glt_backward", &glt_backward, "GLT-Linear backward (scatter-add + STE)");
    m.def("ssm_scan_forward", &ssm_scan_forward, "Fused SSM scan forward");
    m.def("ssm_scan_backward", &ssm_scan_backward, "SSM scan backward (reverse scan)");
    m.def("hadamard_forward", &hadamard_forward, "Fast Walsh-Hadamard forward");
    m.def("hadamard_backward", &hadamard_backward, "Walsh-Hadamard backward");
}
