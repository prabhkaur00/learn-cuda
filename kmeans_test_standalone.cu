// =============================================================================
// kmeans_test_standalone.cu
//
// Self-contained test harness for the 2-way GPU k-means split kernel.
// No external headers needed — GpuSplitResult and gpu_split_kmeans are
// defined right here. Compile and run entirely in one file.
//
// Compile:
//   nvcc -O2 -arch=sm_80 kmeans_test_standalone.cu -o kmeans_test
//
// Run:
//   ./kmeans_test                        # all default tests
//   ./kmeans_test <n> <dim> <max_iters>  # custom single test
// =============================================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cassert>

// ─────────────────────────────────────────────────────────────────────────────
// Helpers: CUDA error checking
// ─────────────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _e = (call);                                                 \
        if (_e != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                 \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// GpuSplitResult  (replaces kmeans_gpu_split.h)
// ─────────────────────────────────────────────────────────────────────────────
struct GpuSplitResult {
    std::vector<float> centroid_a;   // [dim]
    std::vector<float> centroid_b;   // [dim]
    std::vector<int>   partition;    // [n], values in {0,1}
    int                iters_run = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Kernel tuning constants (must match the implementation below)
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int   WARPS_PER_BLOCK = 4;
static constexpr int   THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;   // 128
static constexpr float PERTURB_EPS = 1e-4f;


// =============================================================================
// KERNELS (identical to original — not modified)
// =============================================================================

__global__ static void assign_kernel(
    const float* __restrict__ vecs,
    int n, int dim,
    const float* __restrict__ cA,
    const float* __restrict__ cB,
    const int*   __restrict__ prev_labels,
    int*         __restrict__ labels,
    int*         __restrict__ d_changed)
{
    extern __shared__ float sh[];
    float* sh_cA = sh;
    float* sh_cB = sh + dim;

    const int tid         = (int)threadIdx.x;
    const int lane        = tid & 31;
    const int warp_in_blk = tid >> 5;

    for (int d = tid; d < dim; d += blockDim.x) {
        sh_cA[d] = cA[d];
        sh_cB[d] = cB[d];
    }
    __syncthreads();

    const int vec_idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_blk;
    if (vec_idx >= n) return;

    float dA = 0.f, dB = 0.f;
    const float4* v4  = reinterpret_cast<const float4*>(vecs + (ptrdiff_t)vec_idx * dim);
    const float4* cA4 = reinterpret_cast<const float4*>(sh_cA);
    const float4* cB4 = reinterpret_cast<const float4*>(sh_cB);

    const int n_float4_groups = dim / 128;
    for (int g = 0; g < n_float4_groups; ++g) {
        const int idx4 = g * 32 + lane;
        float4 v  = v4[idx4];
        float4 ca = cA4[idx4];
        float4 cb = cB4[idx4];
        float ex;
        ex = v.x - ca.x; dA += ex * ex;
        ex = v.y - ca.y; dA += ex * ex;
        ex = v.z - ca.z; dA += ex * ex;
        ex = v.w - ca.w; dA += ex * ex;
        float ey;
        ey = v.x - cb.x; dB += ey * ey;
        ey = v.y - cb.y; dB += ey * ey;
        ey = v.z - cb.z; dB += ey * ey;
        ey = v.w - cb.w; dB += ey * ey;
    }

    const unsigned FULL_MASK = 0xffffffffu;
    dA += __shfl_down_sync(FULL_MASK, dA, 16);
    dA += __shfl_down_sync(FULL_MASK, dA,  8);
    dA += __shfl_down_sync(FULL_MASK, dA,  4);
    dA += __shfl_down_sync(FULL_MASK, dA,  2);
    dA += __shfl_down_sync(FULL_MASK, dA,  1);
    dB += __shfl_down_sync(FULL_MASK, dB, 16);
    dB += __shfl_down_sync(FULL_MASK, dB,  8);
    dB += __shfl_down_sync(FULL_MASK, dB,  4);
    dB += __shfl_down_sync(FULL_MASK, dB,  2);
    dB += __shfl_down_sync(FULL_MASK, dB,  1);

    if (lane == 0) {
        const int lbl = (dB < dA) ? 1 : 0;
        labels[vec_idx] = lbl;
        if (prev_labels[vec_idx] != lbl) atomicOr(d_changed, 1);
    }
}

__global__ static void accumulate_kernel(
    const float* __restrict__ vecs,
    int n, int dim,
    const int*   __restrict__ labels,
    float* __restrict__ sumA,
    float* __restrict__ sumB,
    int*   __restrict__ cntA,
    int*   __restrict__ cntB)
{
    extern __shared__ float sh[];
    float* sh_sumA = sh;
    float* sh_sumB = sh + dim;
    int*   sh_cnt  = reinterpret_cast<int*>(sh + 2 * dim);

    const int tid         = (int)threadIdx.x;
    const int lane        = tid & 31;
    const int warp_in_blk = tid >> 5;

    for (int d = tid; d < dim; d += blockDim.x) {
        sh_sumA[d] = 0.f;
        sh_sumB[d] = 0.f;
    }
    if (tid == 0) { sh_cnt[0] = 0; sh_cnt[1] = 0; }
    __syncthreads();

    const int vec_idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_blk;
    if (vec_idx < n) {
        const int lbl    = labels[vec_idx];
        float* sh_target = (lbl == 0) ? sh_sumA : sh_sumB;
        const float4* v4 = reinterpret_cast<const float4*>(
                               vecs + (ptrdiff_t)vec_idx * dim);
        const int n_float4_groups = dim / 128;

        for (int g = 0; g < n_float4_groups; ++g) {
            const int idx4 = g * 32 + lane;
            float4 chunk   = v4[idx4];
            const int base = g * 128 + lane * 4;
            atomicAdd(&sh_target[base + 0], chunk.x);
            atomicAdd(&sh_target[base + 1], chunk.y);
            atomicAdd(&sh_target[base + 2], chunk.z);
            atomicAdd(&sh_target[base + 3], chunk.w);
        }
        if (lane == 0) atomicAdd(&sh_cnt[lbl], 1);
    }

    __syncthreads();
    if (tid == 0) {
        for (int d = 0; d < dim; ++d) {
            if (sh_sumA[d] != 0.f) atomicAdd(&sumA[d], sh_sumA[d]);
            if (sh_sumB[d] != 0.f) atomicAdd(&sumB[d], sh_sumB[d]);
        }
        if (sh_cnt[0] > 0) atomicAdd(cntA, sh_cnt[0]);
        if (sh_cnt[1] > 0) atomicAdd(cntB, sh_cnt[1]);
    }
}

__global__ static void update_kernel(
    const float* __restrict__ sumA,
    const float* __restrict__ sumB,
    const int*   __restrict__ d_cntA,
    const int*   __restrict__ d_cntB,
    float* __restrict__ cA,
    float* __restrict__ cB,
    int dim)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    const float inv_a = 1.f / (float)(*d_cntA);
    const float inv_b = 1.f / (float)(*d_cntB);
    cA[d] = sumA[d] * inv_a;
    cB[d] = sumB[d] * inv_b;
}

__global__ static void seed_centroids_kernel(
    const float* __restrict__ vecs,
    int n, int dim,
    float* __restrict__ cA,
    float* __restrict__ cB,
    float eps)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    float mean = 0.f;
    for (int i = 0; i < n; ++i) mean += vecs[(ptrdiff_t)i * dim + d];
    mean /= (float)n;
    const float perturb = (d & 1) ? eps : -eps;
    cA[d] = mean + perturb;
    cB[d] = mean - perturb;
}


// =============================================================================
// gpu_split_kmeans — library function (unchanged from original)
// =============================================================================
GpuSplitResult gpu_split_kmeans(const float* d_vecs,
                                 int n, int dim, int max_iters)
{
    GpuSplitResult result;
    result.centroid_a.resize(dim, 0.f);
    result.centroid_b.resize(dim, 0.f);
    result.partition.resize(n, 0);
    result.iters_run = 0;

    if (n <= 0 || dim <= 0) return result;

    if (n == 1) {
        cudaMemcpy(result.centroid_a.data(), d_vecs,
                   (size_t)dim * sizeof(float), cudaMemcpyDeviceToHost);
        result.centroid_b = result.centroid_a;
        return result;
    }

    const size_t cen_bytes = (size_t)dim * sizeof(float);
    const size_t lbl_bytes = (size_t)n   * sizeof(int);

    float *d_cA, *d_cB, *d_sumA, *d_sumB;
    int   *d_cntA, *d_cntB, *d_labels, *d_prev_labels, *d_changed;

    CUDA_CHECK(cudaMalloc(&d_cA,          cen_bytes));
    CUDA_CHECK(cudaMalloc(&d_cB,          cen_bytes));
    CUDA_CHECK(cudaMalloc(&d_sumA,        cen_bytes));
    CUDA_CHECK(cudaMalloc(&d_sumB,        cen_bytes));
    CUDA_CHECK(cudaMalloc(&d_cntA,        sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cntB,        sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels,      lbl_bytes));
    CUDA_CHECK(cudaMalloc(&d_prev_labels, lbl_bytes));
    CUDA_CHECK(cudaMalloc(&d_changed,     sizeof(int)));

    {
        const int seed_block = std::min(dim, 256);
        const int seed_grid  = (dim + seed_block - 1) / seed_block;
        seed_centroids_kernel<<<seed_grid, seed_block>>>(
            d_vecs, n, dim, d_cA, d_cB, PERTURB_EPS);
    }

    CUDA_CHECK(cudaMemset(d_prev_labels, 0xff, lbl_bytes));

    const int    assign_block   = THREADS_PER_BLOCK;
    const int    assign_grid    = (n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const size_t smem_assign    = (size_t)2 * dim * sizeof(float);
    const size_t smem_accum     = (size_t)2 * dim * sizeof(float)
                                + (size_t)2 * sizeof(int);
    const int    update_block   = 256;
    const int    update_grid    = (dim + update_block - 1) / update_block;

    for (int iter = 0; iter < max_iters; ++iter) {
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &zero, sizeof(int), cudaMemcpyHostToDevice));

        assign_kernel<<<assign_grid, assign_block, smem_assign>>>(
            d_vecs, n, dim, d_cA, d_cB,
            d_prev_labels, d_labels, d_changed);

        int changed = 0;
        CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        ++result.iters_run;

        CUDA_CHECK(cudaMemcpy(d_prev_labels, d_labels, lbl_bytes, cudaMemcpyDeviceToDevice));
        if (!changed && iter > 0) break;

        CUDA_CHECK(cudaMemset(d_sumA, 0, cen_bytes));
        CUDA_CHECK(cudaMemset(d_sumB, 0, cen_bytes));
        CUDA_CHECK(cudaMemcpy(d_cntA, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cntB, &zero, sizeof(int), cudaMemcpyHostToDevice));

        accumulate_kernel<<<assign_grid, assign_block, smem_accum>>>(
            d_vecs, n, dim, d_labels,
            d_sumA, d_sumB, d_cntA, d_cntB);

        int cntA_h = 0, cntB_h = 0;
        CUDA_CHECK(cudaMemcpy(&cntA_h, d_cntA, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&cntB_h, d_cntB, sizeof(int), cudaMemcpyDeviceToHost));
        if (cntA_h == 0 || cntB_h == 0) break;

        update_kernel<<<update_grid, update_block>>>(
            d_sumA, d_sumB, d_cntA, d_cntB, d_cA, d_cB, dim);
    }

    CUDA_CHECK(cudaMemcpy(result.partition.data(),  d_labels, lbl_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.centroid_a.data(), d_cA,     cen_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.centroid_b.data(), d_cB,     cen_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_cA);  cudaFree(d_cB);
    cudaFree(d_sumA); cudaFree(d_sumB);
    cudaFree(d_cntA); cudaFree(d_cntB);
    cudaFree(d_labels); cudaFree(d_prev_labels);
    cudaFree(d_changed);

    return result;
}


// =============================================================================
// INSTRUMENTED VERSION — per-iteration monitoring
// Wraps gpu_split_kmeans but tracks convergence, sizes, timing per iter.
// =============================================================================
struct IterStats {
    int   iter;
    int   cntA;
    int   cntB;
    float centroid_dist;   // L2 distance between the two centroids
    float elapsed_ms;      // wall-clock time for this iteration
    bool  converged;
};

struct DetailedResult {
    GpuSplitResult         final;
    std::vector<IterStats> stats;
    float                  total_ms;
};

// Helper: L2 distance between two host vectors
static float l2_dist(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.f;
    for (size_t i = 0; i < a.size(); ++i) { float d = a[i]-b[i]; s += d*d; }
    return std::sqrt(s);
}

DetailedResult gpu_split_kmeans_instrumented(const float* d_vecs,
                                              int n, int dim, int max_iters)
{
    DetailedResult dr;
    dr.final.centroid_a.resize(dim, 0.f);
    dr.final.centroid_b.resize(dim, 0.f);
    dr.final.partition.resize(n, 0);
    dr.final.iters_run = 0;
    dr.total_ms = 0.f;

    if (n <= 0 || dim <= 0) return dr;

    const size_t cen_bytes = (size_t)dim * sizeof(float);
    const size_t lbl_bytes = (size_t)n   * sizeof(int);

    float *d_cA, *d_cB, *d_sumA, *d_sumB;
    int   *d_cntA, *d_cntB, *d_labels, *d_prev_labels, *d_changed;

    CUDA_CHECK(cudaMalloc(&d_cA,          cen_bytes));
    CUDA_CHECK(cudaMalloc(&d_cB,          cen_bytes));
    CUDA_CHECK(cudaMalloc(&d_sumA,        cen_bytes));
    CUDA_CHECK(cudaMalloc(&d_sumB,        cen_bytes));
    CUDA_CHECK(cudaMalloc(&d_cntA,        sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cntB,        sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels,      lbl_bytes));
    CUDA_CHECK(cudaMalloc(&d_prev_labels, lbl_bytes));
    CUDA_CHECK(cudaMalloc(&d_changed,     sizeof(int)));

    {
        const int sb = std::min(dim, 256);
        seed_centroids_kernel<<<(dim+sb-1)/sb, sb>>>(
            d_vecs, n, dim, d_cA, d_cB, PERTURB_EPS);
    }
    CUDA_CHECK(cudaMemset(d_prev_labels, 0xff, lbl_bytes));

    const int    AB   = THREADS_PER_BLOCK;
    const int    AG   = (n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const size_t SA   = (size_t)2 * dim * sizeof(float);
    const size_t SC   = SA + (size_t)2 * sizeof(int);
    const int    UB   = 256;
    const int    UG   = (dim + UB - 1) / UB;

    // Host-side centroid copies for distance tracking
    std::vector<float> h_cA(dim), h_cB(dim);

    auto wall = std::chrono::high_resolution_clock::now;
    auto t0   = wall();

    for (int iter = 0; iter < max_iters; ++iter) {
        auto iter_start = wall();

        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &zero, sizeof(int), cudaMemcpyHostToDevice));

        assign_kernel<<<AG, AB, SA>>>(
            d_vecs, n, dim, d_cA, d_cB,
            d_prev_labels, d_labels, d_changed);
        CUDA_CHECK(cudaDeviceSynchronize());

        int changed = 0;
        CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        ++dr.final.iters_run;

        CUDA_CHECK(cudaMemcpy(d_prev_labels, d_labels, lbl_bytes, cudaMemcpyDeviceToDevice));

        bool done = (!changed && iter > 0);

        int cntA_h = 0, cntB_h = 0;
        if (!done) {
            CUDA_CHECK(cudaMemset(d_sumA, 0, cen_bytes));
            CUDA_CHECK(cudaMemset(d_sumB, 0, cen_bytes));
            CUDA_CHECK(cudaMemcpy(d_cntA, &zero, sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_cntB, &zero, sizeof(int), cudaMemcpyHostToDevice));

            accumulate_kernel<<<AG, AB, SC>>>(
                d_vecs, n, dim, d_labels,
                d_sumA, d_sumB, d_cntA, d_cntB);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(&cntA_h, d_cntA, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&cntB_h, d_cntB, sizeof(int), cudaMemcpyDeviceToHost));
            if (cntA_h == 0 || cntB_h == 0) { done = true; }
            else {
                update_kernel<<<UG, UB>>>(
                    d_sumA, d_sumB, d_cntA, d_cntB, d_cA, d_cB, dim);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        } else {
            // Reconstruct counts from labels for reporting
            std::vector<int> h_lbl(n);
            CUDA_CHECK(cudaMemcpy(h_lbl.data(), d_labels, lbl_bytes, cudaMemcpyDeviceToHost));
            for (int x : h_lbl) { if (x==0) ++cntA_h; else ++cntB_h; }
        }

        // Capture centroid distance on host
        CUDA_CHECK(cudaMemcpy(h_cA.data(), d_cA, cen_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cB.data(), d_cB, cen_bytes, cudaMemcpyDeviceToHost));

        auto iter_end = wall();
        float iter_ms = std::chrono::duration<float, std::milli>(iter_end - iter_start).count();

        IterStats s;
        s.iter          = iter + 1;
        s.cntA          = cntA_h;
        s.cntB          = cntB_h;
        s.centroid_dist = l2_dist(h_cA, h_cB);
        s.elapsed_ms    = iter_ms;
        s.converged     = done;
        dr.stats.push_back(s);

        if (done) break;
    }

    dr.total_ms = std::chrono::duration<float, std::milli>(wall() - t0).count();

    CUDA_CHECK(cudaMemcpy(dr.final.partition.data(),  d_labels, lbl_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dr.final.centroid_a.data(), d_cA,     cen_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dr.final.centroid_b.data(), d_cB,     cen_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_cA);  cudaFree(d_cB);
    cudaFree(d_sumA); cudaFree(d_sumB);
    cudaFree(d_cntA); cudaFree(d_cntB);
    cudaFree(d_labels); cudaFree(d_prev_labels);
    cudaFree(d_changed);

    return dr;
}


// =============================================================================
// GPU info printer
// =============================================================================
static void print_gpu_info() {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  GPU: %-47s║\n", p.name);
    printf("║  SM count   : %-38d║\n", p.multiProcessorCount);
    printf("║  Global mem : %-35.1f GB║\n", p.totalGlobalMem / 1e9);
    printf("║  Shared/blk : %-35zu KB║\n", p.sharedMemPerBlock / 1024);
    printf("║  Max threads/blk: %-34d║\n", p.maxThreadsPerBlock);
    printf("║  Compute cap: %d.%-36d║\n", p.major, p.minor);
    printf("╚══════════════════════════════════════════════════════╝\n\n");
}


// =============================================================================
// Data generators
// =============================================================================

// Two well-separated Gaussian blobs
static std::vector<float> make_two_blobs(int n, int dim,
                                          float sep = 5.0f, unsigned seed = 42) {
    std::vector<float> data(n * dim);
    srand(seed);
    for (int i = 0; i < n; ++i) {
        float offset = (i < n / 2) ? -sep / 2.f : sep / 2.f;
        for (int d = 0; d < dim; ++d) {
            float r = ((float)rand() / RAND_MAX - 0.5f) * 2.f;  // [-1, 1]
            data[i * dim + d] = offset + r;
        }
    }
    return data;
}

// Single Gaussian blob (hard case — separation should be small)
static std::vector<float> make_one_blob(int n, int dim, unsigned seed = 7) {
    std::vector<float> data(n * dim);
    srand(seed);
    for (int i = 0; i < n; ++i)
        for (int d = 0; d < dim; ++d)
            data[i * dim + d] = ((float)rand() / RAND_MAX - 0.5f) * 2.f;
    return data;
}

// Worst case: all vectors identical
static std::vector<float> make_identical(int n, int dim) {
    std::vector<float> data(n * dim, 1.0f);
    return data;
}


// =============================================================================
// Printing helpers
// =============================================================================

static void print_iter_table(const std::vector<IterStats>& stats) {
    printf("  %-6s %-10s %-10s %-14s %-12s %s\n",
           "Iter", "Cluster A", "Cluster B", "Centroid Dist", "Time (ms)", "Converged?");
    printf("  %s\n", std::string(64, '-').c_str());
    for (auto& s : stats) {
        printf("  %-6d %-10d %-10d %-14.4f %-12.3f %s\n",
               s.iter, s.cntA, s.cntB,
               s.centroid_dist, s.elapsed_ms,
               s.converged ? "YES ✓" : "no");
    }
}

static void print_result_summary(const DetailedResult& dr, int n, int dim) {
    int cntA = (int)std::count(dr.final.partition.begin(), dr.final.partition.end(), 0);
    int cntB = n - cntA;
    printf("  Iterations run   : %d\n", dr.final.iters_run);
    printf("  Final cluster A  : %d  (%.1f%%)\n", cntA, 100.f * cntA / n);
    printf("  Final cluster B  : %d  (%.1f%%)\n", cntB, 100.f * cntB / n);
    printf("  Centroid dist    : %.6f\n", l2_dist(dr.final.centroid_a, dr.final.centroid_b));
    printf("  Total wall time  : %.3f ms\n", dr.total_ms);
    printf("  Throughput       : %.2f M vectors/s\n", (n / 1e6f) / (dr.total_ms / 1000.f));
}

// Verify: each cluster mean should be close to its centroid
static bool verify_centroids(const DetailedResult& dr, const std::vector<float>& h_vecs,
                              int n, int dim, float tol = 1e-2f) {
    std::vector<double> meanA(dim, 0.0), meanB(dim, 0.0);
    int cA = 0, cB = 0;
    for (int i = 0; i < n; ++i) {
        int lbl = dr.final.partition[i];
        auto& m = (lbl == 0) ? meanA : meanB;
        if (lbl == 0) ++cA; else ++cB;
        for (int d = 0; d < dim; ++d)
            m[d] += h_vecs[i * dim + d];
    }
    if (cA == 0 || cB == 0) {
        printf("  [WARN] Degenerate split — one empty cluster.\n");
        return false;
    }
    for (int d = 0; d < dim; ++d) { meanA[d] /= cA; meanB[d] /= cB; }

    float errA = 0.f, errB = 0.f;
    for (int d = 0; d < dim; ++d) {
        float dA = (float)meanA[d] - dr.final.centroid_a[d];
        float dB = (float)meanB[d] - dr.final.centroid_b[d];
        errA += dA * dA;
        errB += dB * dB;
    }
    errA = std::sqrt(errA / dim);
    errB = std::sqrt(errB / dim);
    printf("  Centroid error (A): %.2e  (B): %.2e  [tol=%.0e]\n", errA, errB, tol);
    bool ok = (errA < tol && errB < tol);
    printf("  Correctness check : %s\n", ok ? "PASS ✓" : "FAIL ✗");
    return ok;
}


// =============================================================================
// Individual test cases
// =============================================================================

static void test_two_blobs(int n, int dim, int max_iters) {
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("TEST: Two Gaussian Blobs  n=%d  dim=%d  max_iters=%d\n", n, dim, max_iters);
    printf("  Expected: balanced split (~50/50), fast convergence\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    auto h_vecs = make_two_blobs(n, dim);
    float* d_vecs;
    CUDA_CHECK(cudaMalloc(&d_vecs, (size_t)n * dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_vecs, h_vecs.data(),
                          (size_t)n * dim * sizeof(float), cudaMemcpyHostToDevice));

    auto dr = gpu_split_kmeans_instrumented(d_vecs, n, dim, max_iters);

    print_iter_table(dr.stats);
    printf("\n");
    print_result_summary(dr, n, dim);
    verify_centroids(dr, h_vecs, n, dim);

    cudaFree(d_vecs);
}

static void test_one_blob(int n, int dim, int max_iters) {
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("TEST: Single Gaussian Blob  n=%d  dim=%d  max_iters=%d\n", n, dim, max_iters);
    printf("  Expected: roughly 50/50 but may oscillate, converges\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    auto h_vecs = make_one_blob(n, dim);
    float* d_vecs;
    CUDA_CHECK(cudaMalloc(&d_vecs, (size_t)n * dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_vecs, h_vecs.data(),
                          (size_t)n * dim * sizeof(float), cudaMemcpyHostToDevice));

    auto dr = gpu_split_kmeans_instrumented(d_vecs, n, dim, max_iters);

    print_iter_table(dr.stats);
    printf("\n");
    print_result_summary(dr, n, dim);
    verify_centroids(dr, h_vecs, n, dim);

    cudaFree(d_vecs);
}

static void test_identical_vectors(int n, int dim) {
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("TEST: Identical Vectors (edge case)  n=%d  dim=%d\n", n, dim);
    printf("  Expected: degenerate split detected, early stop\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    auto h_vecs = make_identical(n, dim);
    float* d_vecs;
    CUDA_CHECK(cudaMalloc(&d_vecs, (size_t)n * dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_vecs, h_vecs.data(),
                          (size_t)n * dim * sizeof(float), cudaMemcpyHostToDevice));

    auto dr = gpu_split_kmeans_instrumented(d_vecs, n, dim, 50);
    print_iter_table(dr.stats);
    printf("\n");
    print_result_summary(dr, n, dim);

    cudaFree(d_vecs);
}

static void test_scaling(int dim, int max_iters) {
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("TEST: Scaling  dim=%d  max_iters=%d\n", dim, max_iters);
    printf("  Measures throughput across different n values\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  %-12s %-8s %-12s %-12s\n", "n", "iters", "time(ms)", "Mvecs/s");
    printf("  %s\n", std::string(48, '-').c_str());

    for (int n : {1000, 10000, 100000, 500000, 1000000}) {
        auto h_vecs = make_two_blobs(n, dim);
        float* d_vecs;
        CUDA_CHECK(cudaMalloc(&d_vecs, (size_t)n * dim * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_vecs, h_vecs.data(),
                              (size_t)n * dim * sizeof(float), cudaMemcpyHostToDevice));

        auto dr = gpu_split_kmeans_instrumented(d_vecs, n, dim, max_iters);

        printf("  %-12d %-8d %-12.2f %-12.2f\n",
               n, dr.final.iters_run, dr.total_ms,
               (n / 1e6f) / (dr.total_ms / 1000.f));

        cudaFree(d_vecs);
    }
}

static void test_dim_scaling(int n, int max_iters) {
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("TEST: Dim Scaling  n=%d  max_iters=%d\n", n, max_iters);
    printf("  Note: dim must be divisible by 128 (float4 × 32 lanes)\n");
    printf("  %-8s %-8s %-12s %-12s\n", "dim", "iters", "time(ms)", "Mvecs/s");
    printf("  %s\n", std::string(44, '-').c_str());

    for (int dim : {128, 256, 512, 1024}) {
        auto h_vecs = make_two_blobs(n, dim);
        float* d_vecs;
        CUDA_CHECK(cudaMalloc(&d_vecs, (size_t)n * dim * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_vecs, h_vecs.data(),
                              (size_t)n * dim * sizeof(float), cudaMemcpyHostToDevice));

        auto dr = gpu_split_kmeans_instrumented(d_vecs, n, dim, max_iters);

        printf("  %-8d %-8d %-12.2f %-12.2f\n",
               dim, dr.final.iters_run, dr.total_ms,
               (n / 1e6f) / (dr.total_ms / 1000.f));

        cudaFree(d_vecs);
    }
}


// =============================================================================
// main
// =============================================================================
int main(int argc, char* argv[]) {
    printf("\n");
    printf("  ██╗  ██╗███╗   ███╗███████╗ █████╗ ███╗   ██╗███████╗\n");
    printf("  ██║ ██╔╝████╗ ████║██╔════╝██╔══██╗████╗  ██║██╔════╝\n");
    printf("  █████╔╝ ██╔████╔██║█████╗  ███████║██╔██╗ ██║███████╗\n");
    printf("  ██╔═██╗ ██║╚██╔╝██║██╔══╝  ██╔══██║██║╚██╗██║╚════██║\n");
    printf("  ██║  ██╗██║ ╚═╝ ██║███████╗██║  ██║██║ ╚████║███████║\n");
    printf("  ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝\n");
    printf("  GPU 2-way K-Means Split — Test Harness\n\n");

    print_gpu_info();

    if (argc == 4) {
        // Custom single test mode
        int n         = atoi(argv[1]);
        int dim       = atoi(argv[2]);
        int max_iters = atoi(argv[3]);

        if (dim % 128 != 0) {
            fprintf(stderr, "ERROR: dim must be divisible by 128 (got %d)\n", dim);
            return 1;
        }
        test_two_blobs(n, dim, max_iters);
        return 0;
    }

    // ── Default test suite ───────────────────────────────────────────────────
    // 1. Core correctness on two well-separated blobs
    test_two_blobs(10000, 128, 100);

    // 2. Harder: single blob (tests convergence with small centroid movement)
    test_one_blob(10000, 128, 100);

    // 3. Edge case: identical vectors
    test_identical_vectors(1024, 128);

    // 4. Scale across n (fixed dim=128)
    test_scaling(128, 50);

    // 5. Scale across dim (fixed n=50000)
    test_dim_scaling(50000, 50);

    printf("\n══════════════════════════════════════════════════════\n");
    printf("  All tests complete.\n");
    printf("══════════════════════════════════════════════════════\n\n");

    return 0;
}
