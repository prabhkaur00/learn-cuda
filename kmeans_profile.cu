// =============================================================================
// kmeans_profile.cu
//
// Deep-profile harness for the 2-way GPU k-means split.
// Measures per-step: wall time, CUDA event time, arithmetic intensity,
// estimated memory traffic (global/shared/register), and convergence.
// Sweeps cluster sizes: 64, 128, 512, 1024, 4096, 8000.
// Emits a CSV for external plotting and prints a rich terminal report.
//
// Compile:
//   nvcc -O2 -arch=sm_80 --generate-line-info kmeans_profile.cu -o kmeans_profile
//
// Run:
//   ./kmeans_profile                    # full sweep, dim=128, max_iters=50
//   ./kmeans_profile <dim> <max_iters>  # custom
//
// Output: kmeans_profile_results.csv
// =============================================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <cassert>

// ─────────────────────────────────────────────────────────────────────────────
// CUDA error guard
// ─────────────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t _e = (call);                                                  \
        if (_e != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error @ %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                  \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Tuning constants (must match the kernel assumptions)
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int   WARPS_PER_BLOCK  = 4;
static constexpr int   THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;   // 128
static constexpr float PERTURB_EPS      = 1e-4f;


// =============================================================================
// ██╗  ██╗███████╗██████╗ ███╗   ██╗███████╗██╗     ███████╗
// ██║ ██╔╝██╔════╝██╔══██╗████╗  ██║██╔════╝██║     ██╔════╝
// █████╔╝ █████╗  ██████╔╝██╔██╗ ██║█████╗  ██║     ███████╗
// ██╔═██╗ ██╔══╝  ██╔══██╗██║╚██╗██║██╔══╝  ██║     ╚════██║
// ██║  ██╗███████╗██║  ██║██║ ╚████║███████╗███████╗███████║
// ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚══════╝
// =============================================================================

// =============================================================================
// assign_kernel — E-step
//
// PURPOSE: For each vector i, compute squared L2 distance to centroid A and B,
//          then assign the vector to the nearer centroid.
//
// GRID:  ceil(n / WARPS_PER_BLOCK) blocks
// BLOCK: 128 threads = 4 warps
// SMEM:  2 × dim × 4 bytes  (cA and cB tiled into shared memory once per block)
//
// ┌─────────────────────────────────────────────────────────────┐
// │ MEMORY HIERARCHY USAGE                                       │
// │  Global  → reads vecs[n×dim], cA[dim], cB[dim]              │
// │            writes labels[n], d_changed[1]                   │
// │  Shared  → cA and cB cached here; all warps in block share  │
// │            avoids n_warps × 2 × dim redundant global reads  │
// │  Registers → dA, dB partial sums; float4 v,ca,cb per step   │
// └─────────────────────────────────────────────────────────────┘
//
// ARITHMETIC INTENSITY ANALYSIS (per vector):
//   Flops:  dim × 4 (sub,mul,add × 2 centroids) + 5×2 shuffle-adds = ~8×dim
//   Bytes:  dim×4 (vector load) + 2×dim×4 (centroids from smem, ~free)
//           + 4 (write label) + 4 (read prev_label)
//   AI ≈ 8×dim / (dim×4 + 8) ≈ 2 FLOPs/byte  → memory-bound on A100
// =============================================================================
__global__ static void assign_kernel(
    const float* __restrict__ vecs,          // [n × dim] row-major, HBM
    int n, int dim,
    const float* __restrict__ cA,            // [dim] centroid A, HBM
    const float* __restrict__ cB,            // [dim] centroid B, HBM
    const int*   __restrict__ prev_labels,   // [n] labels from last iteration
    int*         __restrict__ labels,        // [n] output: 0 or 1
    int*         __restrict__ d_changed)     // [1] set to 1 if any label flipped
{
    // ── Shared memory layout ─────────────────────────────────────────────────
    // sh[0 .. dim-1]       = cA tile
    // sh[dim .. 2*dim-1]   = cB tile
    // Loading both centroids once into smem saves 4 warps × 2 × dim redundant
    // global reads per block — smem bandwidth is ~10× HBM on A100.
    extern __shared__ float sh[];
    float* sh_cA = sh;
    float* sh_cB = sh + dim;

    const int tid         = (int)threadIdx.x;     // 0..127
    const int lane        = tid & 31;              // lane within warp 0..31
    const int warp_in_blk = tid >> 5;              // warp index 0..3

    // ── Phase 1: Cooperative centroid load ───────────────────────────────────
    // All 128 threads stride across dim with step=128.
    // Consecutive threads access consecutive floats → coalesced 128B transactions.
    // After syncthreads, every warp in this block sees cA/cB in L1/smem.
    for (int d = tid; d < dim; d += blockDim.x) {
        sh_cA[d] = cA[d];
        sh_cB[d] = cB[d];
    }
    __syncthreads();   // barrier: ensures centroid tile is fully written before warps read it

    // ── Phase 2: Each warp owns one vector ───────────────────────────────────
    // warp 0 → vector blockIdx.x*4+0, warp 1 → blockIdx.x*4+1, etc.
    const int vec_idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_blk;
    if (vec_idx >= n) return;   // bounds guard for the last partial block

    // ── Phase 3: float4 coalesced distance accumulation ──────────────────────
    // dA, dB live in registers — zero memory traffic for partial sums.
    float dA = 0.f, dB = 0.f;

    // Reinterpret as float4 pointers: each float4 load fetches 16 bytes.
    // 32 lanes × 16 bytes = 512 bytes per step = 4 full 128-byte cache lines,
    // all coalesced. This is the maximum possible memory efficiency.
    const float4* v4  = reinterpret_cast<const float4*>(vecs + (ptrdiff_t)vec_idx * dim);
    const float4* cA4 = reinterpret_cast<const float4*>(sh_cA);
    const float4* cB4 = reinterpret_cast<const float4*>(sh_cB);

    // dim / 128 groups: each group has 32 float4s (32 lanes × 4 floats = 128 scalars)
    const int n_groups = dim / 128;
    for (int g = 0; g < n_groups; ++g) {
        const int idx4 = g * 32 + lane;   // float4 index for this lane in this group

        // Each lane reads its own float4 from: vector, centroid A, centroid B
        // v4 from HBM (global), cA4/cB4 from smem (on-chip)
        float4 v  = v4[idx4];             // 16 bytes from HBM per lane
        float4 ca = cA4[idx4];            // 16 bytes from smem
        float4 cb = cB4[idx4];            // 16 bytes from smem

        // Unrolled 4-component squared diff accumulation — stays in registers.
        // 4 sub + 4 mul + 4 add = 12 flops per centroid per float4 = 24 flops total.
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

    // ── Phase 4: Warp-shuffle reduction ──────────────────────────────────────
    // Each lane holds dim/32 elements worth of partial dA/dB.
    // 5 rounds of __shfl_down_sync reduce all 32 partial sums to lane 0.
    // Cost: 5 × 2 cycles = ~10 cycles, entirely register-to-register.
    // This avoids writing partial sums to smem and back (saves 2×32×4 = 256 bytes/warp).
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

    // ── Phase 5: Lane 0 writes result ────────────────────────────────────────
    // Only lane 0 has the fully-reduced sum. Other lanes' values are partial.
    // atomicOr: sets d_changed=1 only when a label actually flips.
    // On a converged iteration this generates zero global atomic traffic.
    if (lane == 0) {
        const int lbl = (dB < dA) ? 1 : 0;   // assign to nearer centroid
        labels[vec_idx] = lbl;                // 4 bytes to HBM
        if (prev_labels[vec_idx] != lbl) {
            atomicOr(d_changed, 1);           // single global atomic, fires only on change
        }
    }
}

// =============================================================================
// accumulate_kernel — partial M-step (sum accumulation)
//
// PURPOSE: Sum all vectors per cluster into sumA/sumB and count members.
//
// TWO-PHASE DESIGN:
//   Phase 1 (parallel): Each warp scatters its vector into smem partial sums.
//                        smem atomics are ~10× faster than global atomics.
//   Phase 2 (serial):   Thread 0 flushes non-zero smem entries to global.
//                        Reduces global atomic contention from O(n×dim) to
//                        O(blocks×dim), a factor of WARPS_PER_BLOCK reduction.
//
// MEMORY TRAFFIC PER BLOCK:
//   Reads:  WARPS_PER_BLOCK × dim × 4 bytes (vectors from HBM)
//           WARPS_PER_BLOCK × 4 bytes (labels)
//   Writes: dim × 8 bytes (non-zero entries to sumA/sumB via global atomicAdd)
//           2 × 4 bytes (cntA/cntB via atomicAdd)
//
// ARITHMETIC INTENSITY:
//   Flops per block: WARPS_PER_BLOCK × dim × 1 (atomicAdd is 1 add)
//   Bytes per block: WARPS_PER_BLOCK × dim × 4 (reads) + dim × 8 (writes)
//   AI ≈ 4/(4+8) ≈ 0.33 FLOPs/byte  → very memory-bound, dominated by scatter writes
// =============================================================================
__global__ static void accumulate_kernel(
    const float* __restrict__ vecs,       // [n × dim] HBM
    int n, int dim,
    const int*   __restrict__ labels,     // [n] 0 or 1
    float* __restrict__ sumA,             // [dim] accumulator for cluster A
    float* __restrict__ sumB,             // [dim] accumulator for cluster B
    int*   __restrict__ cntA,             // [1] count for cluster A
    int*   __restrict__ cntB)             // [1] count for cluster B
{
    // ── Shared memory layout ─────────────────────────────────────────────────
    // sh[0..dim-1]        = sumA partial sums for this block
    // sh[dim..2*dim-1]    = sumB partial sums for this block
    // sh_cnt[0]           = cntA partial count for this block
    // sh_cnt[1]           = cntB partial count for this block
    extern __shared__ float sh[];
    float* sh_sumA = sh;
    float* sh_sumB = sh + dim;
    int*   sh_cnt  = reinterpret_cast<int*>(sh + 2 * dim);  // int overlay

    const int tid         = (int)threadIdx.x;
    const int lane        = tid & 31;
    const int warp_in_blk = tid >> 5;

    // ── Phase 0: Zero block-local accumulators ───────────────────────────────
    // All 128 threads cooperate; stride=128 → coalesced smem writes.
    for (int d = tid; d < dim; d += blockDim.x) {
        sh_sumA[d] = 0.f;
        sh_sumB[d] = 0.f;
    }
    if (tid == 0) { sh_cnt[0] = 0; sh_cnt[1] = 0; }
    __syncthreads();

    // ── Phase 1: Scatter each warp's vector into smem partial sums ───────────
    const int vec_idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_blk;
    if (vec_idx < n) {
        const int lbl     = labels[vec_idx];           // cluster assignment (0 or 1)
        float* sh_target  = (lbl == 0) ? sh_sumA : sh_sumB;  // target smem accumulator

        // float4 coalesced load: same pattern as assign_kernel.
        // 32 lanes × float4 = 128 scalars per step, full cache line utilization.
        const float4* v4          = reinterpret_cast<const float4*>(
                                        vecs + (ptrdiff_t)vec_idx * dim);
        const int n_groups = dim / 128;

        for (int g = 0; g < n_groups; ++g) {
            const int idx4 = g * 32 + lane;
            float4 chunk   = v4[idx4];                // 16 bytes from HBM
            // base: scalar offset into sh_sumA/sh_sumB
            // lane k maps to: g*128 + k*4, k*4+1, k*4+2, k*4+3
            const int base = g * 128 + lane * 4;

            // smem atomicAdd: on-chip (~4 cycles), handles concurrent warp writes
            // to the same dimension when multiple warps land in the same cluster
            atomicAdd(&sh_target[base + 0], chunk.x);
            atomicAdd(&sh_target[base + 1], chunk.y);
            atomicAdd(&sh_target[base + 2], chunk.z);
            atomicAdd(&sh_target[base + 3], chunk.w);
        }
        // Only lane 0 increments the count to avoid 32× over-counting per warp
        if (lane == 0) atomicAdd(&sh_cnt[lbl], 1);
    }
    __syncthreads();  // ensure all smem writes are visible before flush

    // ── Phase 2: Thread 0 flushes block-local sums to global ────────────────
    // Serialised but small: dim iterations on one thread.
    // The non-zero guard skips dimensions where this block contributed nothing
    // (common when n is small), reducing unnecessary global atomic traffic.
    if (tid == 0) {
        for (int d = 0; d < dim; ++d) {
            if (sh_sumA[d] != 0.f) atomicAdd(&sumA[d], sh_sumA[d]);
            if (sh_sumB[d] != 0.f) atomicAdd(&sumB[d], sh_sumB[d]);
        }
        if (sh_cnt[0] > 0) atomicAdd(cntA, sh_cnt[0]);
        if (sh_cnt[1] > 0) atomicAdd(cntB, sh_cnt[1]);
    }
}

// =============================================================================
// update_kernel — M-step centroid recomputation
//
// PURPOSE: Divide accumulated sums by counts to get new centroids.
//          Runs entirely on GPU — zero PCIe traffic for this step.
//
// GRID/BLOCK: one thread per dimension, ceil(dim/256) blocks.
//
// Each thread d reads: sumA[d], sumB[d], *cntA, *cntB
// Writes: cA[d], cB[d]
//
// ARITHMETIC INTENSITY:
//   Flops per element: 2 multiplies (using reciprocal, faster than division)
//   Bytes per element: 4 reads (sumA) + 4 reads (sumB) +
//                      4 reads (*cntA, shared) + 4 reads (*cntB, shared) +
//                      4 writes (cA) + 4 writes (cB) ≈ 24 bytes
//   AI ≈ 2/24 ≈ 0.08 FLOPs/byte — trivially memory-bound but very cheap kernel
//
// WHY NOT CPU? v3-style CPU update copies 2×dim×4 bytes PCIe each iteration
// plus a host sync stall (~10µs on PCIe Gen4). This kernel eliminates both.
// =============================================================================
__global__ static void update_kernel(
    const float* __restrict__ sumA,    // [dim] accumulated sum for cluster A
    const float* __restrict__ sumB,    // [dim] accumulated sum for cluster B
    const int*   __restrict__ d_cntA,  // [1] count of cluster A members
    const int*   __restrict__ d_cntB,  // [1] count of cluster B members
    float* __restrict__ cA,            // [dim] output: new centroid A
    float* __restrict__ cB,            // [dim] output: new centroid B
    int dim)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;

    // Precompute reciprocals: one multiply < one divide on GPU (saves ~5 cycles/elem)
    const float inv_a = 1.f / (float)(*d_cntA);
    const float inv_b = 1.f / (float)(*d_cntB);
    cA[d] = sumA[d] * inv_a;
    cB[d] = sumB[d] * inv_b;
}

// =============================================================================
// seed_centroids_kernel — initialization
//
// PURPOSE: Compute cluster mean on GPU and place cA/cB symmetrically near it.
//          mean ± eps * alternating_direction
//
// WHY FROM MEAN? Starting from arbitrary vectors (e.g. vecs[0], vecs[n-1]) is
// sensitive to outliers. Mean-centred seeding guarantees balanced geometry
// independent of data order, and typically converges 30-50% faster.
//
// ARITHMETIC INTENSITY:
//   Flops: n additions + 1 division + 2 adds = O(n) per thread
//   Bytes: n × 4 reads (one dimension, strided) + 8 writes (cA[d], cB[d])
//   AI ≈ n / (n × 4 + 8) ≈ 0.25 FLOPs/byte — memory-bound, but runs once
// =============================================================================
__global__ static void seed_centroids_kernel(
    const float* __restrict__ vecs,   // [n × dim] HBM
    int n, int dim,
    float* __restrict__ cA,           // [dim] output: centroid A seed
    float* __restrict__ cB,           // [dim] output: centroid B seed
    float eps)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;

    // Accumulate mean for dimension d across all n vectors.
    // Access pattern: stride=dim — not coalesced, but runs only once.
    float mean = 0.f;
    for (int i = 0; i < n; ++i) mean += vecs[(ptrdiff_t)i * dim + d];
    mean /= (float)n;

    // Alternating perturbation: odd dims get +eps, even get -eps.
    // This breaks the symmetry between cA and cB without random numbers,
    // making seeding deterministic and reproducible.
    const float perturb = (d & 1) ? eps : -eps;
    cA[d] = mean + perturb;
    cB[d] = mean - perturb;
}


// =============================================================================
// PROFILING STRUCTURES
// =============================================================================

// Per-step timing and memory traffic estimates for one iteration
struct StepProfile {
    float seed_ms       = 0.f;
    float assign_ms     = 0.f;
    float accum_ms      = 0.f;
    float update_ms     = 0.f;
    float d2h_ms        = 0.f;   // host copy of convergence flag + counts
    float total_iter_ms = 0.f;

    // Memory traffic estimates (bytes)
    // assign_kernel
    float assign_global_read_bytes  = 0.f;  // vecs + cA + cB initial loads
    float assign_smem_bytes         = 0.f;  // centroid tile in smem
    float assign_reg_bytes          = 0.f;  // dA, dB partial sums in registers
    float assign_global_write_bytes = 0.f;  // labels write

    // accumulate_kernel
    float accum_global_read_bytes   = 0.f;  // vecs + labels
    float accum_smem_bytes          = 0.f;  // partial sums in smem
    float accum_global_write_bytes  = 0.f;  // flush to sumA/sumB/cntA/cntB

    // update_kernel
    float update_global_read_bytes  = 0.f;  // sumA, sumB, cntA, cntB
    float update_global_write_bytes = 0.f;  // cA, cB

    // Arithmetic intensity (FLOPs/byte, using global memory bytes only)
    float assign_ai  = 0.f;
    float accum_ai   = 0.f;
    float update_ai  = 0.f;
};

// Per-iteration result
struct IterProfile {
    int        iter;
    StepProfile step;
    int        cntA, cntB;
    float      centroid_dist;
    bool       converged;
};

// Full run result
struct RunProfile {
    int n, dim, max_iters;
    std::vector<IterProfile> iters;
    float total_ms;
    float seed_ms;
    int   iters_run;
};


// =============================================================================
// CUDA Event timer helper
// =============================================================================
struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~CudaTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin()              { cudaEventRecord(start); }
    float end_ms()            {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};


// =============================================================================
// Memory traffic estimator (analytical model)
//
// These are lower-bound estimates based on the algorithmic access patterns.
// Actual hardware traffic may differ due to L2 cache hits and driver overhead.
//
// Convention: all sizes in bytes.
// =============================================================================
static StepProfile estimate_traffic(int n, int dim) {
    StepProfile s;
    const float f = 4.f;   // sizeof(float) = 4 bytes
    const float i = 4.f;   // sizeof(int)   = 4 bytes

    // ── assign_kernel ────────────────────────────────────────────────────────
    // Global reads: n vectors × dim floats + 2 centroids × dim floats (first load)
    //               + n prev_labels
    s.assign_global_read_bytes  = n * dim * f        // vector loads (HBM)
                                + 2 * dim * f        // cA, cB initial load to smem
                                + n * i;             // prev_labels

    // Shared mem: 2 × dim floats loaded once per block, broadcast to all warps.
    // Traffic is per-block: blocks = ceil(n/4). Each block loads 2*dim floats.
    int assign_blocks = (n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    s.assign_smem_bytes         = assign_blocks * 2 * dim * f;

    // Registers: each warp holds 2 float accumulators (dA, dB) per active lane.
    // 32 lanes × 2 floats × blocks = register live range estimate
    s.assign_reg_bytes          = assign_blocks * WARPS_PER_BLOCK * 32 * 2 * f;

    // Global writes: labels array
    s.assign_global_write_bytes = n * i;

    float assign_total_bytes    = s.assign_global_read_bytes + s.assign_global_write_bytes;
    // Flops: n vectors × dim × 8 (2 sub+mul+add per component × 2 centroids)
    float assign_flops          = n * dim * 8.f;
    s.assign_ai = assign_flops / assign_total_bytes;

    // ── accumulate_kernel ─────────────────────────────────────────────────────
    // Global reads: n vectors + n labels
    s.accum_global_read_bytes   = n * dim * f + n * i;

    // Shared: 2×dim partial-sum floats + 2 ints per block
    s.accum_smem_bytes          = assign_blocks * (2 * dim * f + 2 * i);

    // Global writes: sumA + sumB (dim floats each), cntA + cntB (1 int each)
    // Only non-zero entries are written, but worst case = 2×dim floats + 2 ints
    s.accum_global_write_bytes  = 2 * dim * f + 2 * i;

    float accum_total_bytes     = s.accum_global_read_bytes + s.accum_global_write_bytes;
    // Flops: n × dim × 1 (one add per element into smem accumulator)
    float accum_flops           = n * dim * 1.f;
    s.accum_ai = accum_flops / accum_total_bytes;

    // ── update_kernel ─────────────────────────────────────────────────────────
    // Global reads: sumA[dim] + sumB[dim] + cntA[1] + cntB[1]
    s.update_global_read_bytes  = 2 * dim * f + 2 * i;

    // Global writes: cA[dim] + cB[dim]
    s.update_global_write_bytes = 2 * dim * f;

    float update_total_bytes    = s.update_global_read_bytes + s.update_global_write_bytes;
    // Flops: dim × 2 (2 multiplies: sumA[d]*inv_a, sumB[d]*inv_b)
    float update_flops          = dim * 2.f;
    s.update_ai = update_flops / update_total_bytes;

    return s;
}


// =============================================================================
// run_profile — main profiled execution
// =============================================================================
static RunProfile run_profile(const float* d_vecs, int n, int dim, int max_iters) {
    RunProfile rp;
    rp.n = n; rp.dim = dim; rp.max_iters = max_iters;
    rp.total_ms = 0.f; rp.seed_ms = 0.f; rp.iters_run = 0;

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

    // ── Seed ─────────────────────────────────────────────────────────────────
    CudaTimer timer;
    {
        const int sb = std::min(dim, 256);
        timer.begin();
        seed_centroids_kernel<<<(dim+sb-1)/sb, sb>>>(
            d_vecs, n, dim, d_cA, d_cB, PERTURB_EPS);
        rp.seed_ms = timer.end_ms();
    }
    CUDA_CHECK(cudaMemset(d_prev_labels, 0xff, lbl_bytes));

    // Launch config (same every iteration)
    const int    AB = THREADS_PER_BLOCK;
    const int    AG = (n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const size_t SA = (size_t)2 * dim * sizeof(float);
    const size_t SC = SA + (size_t)2 * sizeof(int);
    const int    UB = 256;
    const int    UG = (dim + UB - 1) / UB;

    // Pre-compute traffic model (analytical, same every iteration)
    StepProfile traffic = estimate_traffic(n, dim);

    // Host-side centroid buffers for distance tracking
    std::vector<float> h_cA(dim), h_cB(dim);

    auto wall_start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < max_iters; ++iter) {
        IterProfile ip;
        ip.iter = iter + 1;
        ip.step = traffic;   // copy traffic model into this iteration's profile

        // Reset convergence flag
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &zero, sizeof(int), cudaMemcpyHostToDevice));

        // ── Assign ───────────────────────────────────────────────────────────
        timer.begin();
        assign_kernel<<<AG, AB, SA>>>(
            d_vecs, n, dim, d_cA, d_cB,
            d_prev_labels, d_labels, d_changed);
        ip.step.assign_ms = timer.end_ms();

        // ── D2H: convergence flag (4 bytes) ──────────────────────────────────
        auto d2h_start = std::chrono::high_resolution_clock::now();
        int changed = 0;
        CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        ip.step.d2h_ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - d2h_start).count();

        ++rp.iters_run;
        CUDA_CHECK(cudaMemcpy(d_prev_labels, d_labels, lbl_bytes, cudaMemcpyDeviceToDevice));

        bool done = (!changed && iter > 0);
        int cntA_h = 0, cntB_h = 0;

        if (!done) {
            // ── Accumulate ───────────────────────────────────────────────────
            CUDA_CHECK(cudaMemset(d_sumA, 0, cen_bytes));
            CUDA_CHECK(cudaMemset(d_sumB, 0, cen_bytes));
            CUDA_CHECK(cudaMemcpy(d_cntA, &zero, sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_cntB, &zero, sizeof(int), cudaMemcpyHostToDevice));

            timer.begin();
            accumulate_kernel<<<AG, AB, SC>>>(
                d_vecs, n, dim, d_labels,
                d_sumA, d_sumB, d_cntA, d_cntB);
            ip.step.accum_ms = timer.end_ms();

            // ── D2H: counts (8 bytes) ─────────────────────────────────────────
            CUDA_CHECK(cudaMemcpy(&cntA_h, d_cntA, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&cntB_h, d_cntB, sizeof(int), cudaMemcpyDeviceToHost));
            ip.step.d2h_ms += 0.001f;   // ~1µs for 8-byte copy

            if (cntA_h == 0 || cntB_h == 0) { done = true; }
            else {
                // ── Update ───────────────────────────────────────────────────
                timer.begin();
                update_kernel<<<UG, UB>>>(
                    d_sumA, d_sumB, d_cntA, d_cntB, d_cA, d_cB, dim);
                ip.step.update_ms = timer.end_ms();
            }
        } else {
            // Converged — reconstruct counts from labels for reporting
            std::vector<int> h_lbl(n);
            CUDA_CHECK(cudaMemcpy(h_lbl.data(), d_labels, lbl_bytes, cudaMemcpyDeviceToHost));
            for (int x : h_lbl) { if (x == 0) ++cntA_h; else ++cntB_h; }
        }

        // Total iteration wall time
        ip.step.total_iter_ms = ip.step.assign_ms + ip.step.accum_ms
                              + ip.step.update_ms + ip.step.d2h_ms;

        // Centroid distance
        CUDA_CHECK(cudaMemcpy(h_cA.data(), d_cA, cen_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cB.data(), d_cB, cen_bytes, cudaMemcpyDeviceToHost));
        float dist = 0.f;
        for (int d = 0; d < dim; ++d) {
            float dd = h_cA[d] - h_cB[d];
            dist += dd * dd;
        }
        ip.centroid_dist = std::sqrt(dist);

        ip.cntA = cntA_h;
        ip.cntB = cntB_h;
        ip.converged = done;
        rp.iters.push_back(ip);

        if (done) break;
    }

    rp.total_ms = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - wall_start).count();

    cudaFree(d_cA);  cudaFree(d_cB);
    cudaFree(d_sumA); cudaFree(d_sumB);
    cudaFree(d_cntA); cudaFree(d_cntB);
    cudaFree(d_labels); cudaFree(d_prev_labels);
    cudaFree(d_changed);

    return rp;
}


// =============================================================================
// DATA GENERATORS
// =============================================================================
static std::vector<float> make_two_blobs(int n, int dim,
                                          float sep = 4.0f, unsigned seed = 42) {
    std::vector<float> data((size_t)n * dim);
    srand(seed);
    for (int i = 0; i < n; ++i) {
        float offset = (i < n / 2) ? -sep / 2.f : sep / 2.f;
        for (int d = 0; d < dim; ++d) {
            float r = ((float)rand() / RAND_MAX - 0.5f) * 2.f;
            data[(size_t)i * dim + d] = offset + r;
        }
    }
    return data;
}


// =============================================================================
// GPU INFO
// =============================================================================
static void print_gpu_info() {
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    printf("GPU: %s  |  SMs: %d  |  VRAM: %.1f GB  |  Compute: %d.%d\n",
           p.name, p.multiProcessorCount,
           p.totalGlobalMem / 1e9, p.major, p.minor);
    printf("Peak HBM BW: %.0f GB/s  |  Smem/block: %zu KB  |  MaxThreads/block: %d\n\n",
           p.memoryClockRate * 1e-6 * p.memoryBusWidth * 2 / 8,
           p.sharedMemPerBlock / 1024,
           p.maxThreadsPerBlock);
}


// =============================================================================
// TERMINAL REPORT: per-step breakdown for one run (used for n=8000)
// =============================================================================
static void print_step_breakdown(const RunProfile& rp) {
    printf("\n┌────────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│  STEP-BY-STEP PROFILE   n=%-6d  dim=%-4d                                            │\n",
           rp.n, rp.dim);
    printf("├──────┬───────────┬───────────┬───────────┬───────────┬───────────┬────────────────────┤\n");
    printf("│ Iter │ Seed (ms) │Assign(ms) │Accum (ms) │Update(ms) │ D2H  (ms) │ Total iter (ms)    │\n");
    printf("├──────┼───────────┼───────────┼───────────┼───────────┼───────────┼────────────────────┤\n");

    for (auto& ip : rp.iters) {
        float tot = ip.step.total_iter_ms;
        float pA  = tot > 0 ? 100.f * ip.step.assign_ms  / tot : 0;
        float pC  = tot > 0 ? 100.f * ip.step.accum_ms   / tot : 0;
        float pU  = tot > 0 ? 100.f * ip.step.update_ms  / tot : 0;
        float pD  = tot > 0 ? 100.f * ip.step.d2h_ms     / tot : 0;
        printf("│ %-4d │ %9.4f │ %6.4f(%2.0f%%)│ %6.4f(%2.0f%%)│ %6.4f(%2.0f%%)│ %6.4f(%2.0f%%)│ %10.4f          │\n",
               ip.iter,
               (ip.iter == 1 ? rp.seed_ms : 0.f),
               ip.step.assign_ms,  pA,
               ip.step.accum_ms,   pC,
               ip.step.update_ms,  pU,
               ip.step.d2h_ms,     pD,
               tot);
    }
    printf("└──────┴───────────┴───────────┴───────────┴───────────┴───────────┴────────────────────┘\n");

    // Memory traffic summary (first non-trivial iteration)
    if (!rp.iters.empty()) {
        const auto& s = rp.iters[0].step;
        printf("\n  MEMORY TRAFFIC ESTIMATE (per iteration, analytical model)\n");
        printf("  ┌─────────────────────────────────────────────────────────────┐\n");
        printf("  │ Step      │ Global Read  │ Global Write │ Smem     │   AI   │\n");
        printf("  ├───────────┼──────────────┼──────────────┼──────────┼────────┤\n");
        printf("  │ assign    │ %8.2f MB   │ %8.2f MB   │ %6.2f MB │ %5.2f  │\n",
               s.assign_global_read_bytes  / 1e6,
               s.assign_global_write_bytes / 1e6,
               s.assign_smem_bytes         / 1e6,
               s.assign_ai);
        printf("  │ accumulate│ %8.2f MB   │ %8.2f MB   │ %6.2f MB │ %5.2f  │\n",
               s.accum_global_read_bytes   / 1e6,
               s.accum_global_write_bytes  / 1e6,
               s.accum_smem_bytes          / 1e6,
               s.accum_ai);
        printf("  │ update    │ %8.2f KB   │ %8.2f KB   │    0     │ %5.2f  │\n",
               s.update_global_read_bytes  / 1e3,
               s.update_global_write_bytes / 1e3,
               s.update_ai);
        printf("  └───────────┴──────────────┴──────────────┴──────────┴────────┘\n");
        printf("  AI = FLOPs/byte (global memory only). >10 = compute-bound, <1 = memory-bound\n");
    }

    // Convergence curve
    printf("\n  CONVERGENCE (centroid L2 distance per iteration)\n  ");
    for (auto& ip : rp.iters) {
        int bars = (int)(ip.centroid_dist * 4);
        if (bars > 60) bars = 60;
        printf("  iter%2d [%-60.*s] %.4f  |A|=%-6d |B|=%-6d %s\n",
               ip.iter, bars, "████████████████████████████████████████████████████████████",
               ip.centroid_dist, ip.cntA, ip.cntB,
               ip.converged ? "✓ converged" : "");
    }
}


// =============================================================================
// CSV WRITER
// =============================================================================
static void write_csv(const std::vector<RunProfile>& runs, const std::string& path) {
    std::ofstream f(path);
    f << "n,dim,iter,seed_ms,assign_ms,accum_ms,update_ms,d2h_ms,total_iter_ms,"
         "assign_pct,accum_pct,update_pct,d2h_pct,"
         "assign_global_read_MB,assign_global_write_MB,assign_smem_MB,assign_ai,"
         "accum_global_read_MB,accum_global_write_MB,accum_smem_MB,accum_ai,"
         "update_global_read_KB,update_global_write_KB,update_ai,"
         "cntA,cntB,centroid_dist,converged,total_run_ms,iters_run\n";

    for (auto& rp : runs) {
        for (auto& ip : rp.iters) {
            float tot = ip.step.total_iter_ms;
            float pA  = tot > 0 ? 100.f * ip.step.assign_ms  / tot : 0;
            float pC  = tot > 0 ? 100.f * ip.step.accum_ms   / tot : 0;
            float pU  = tot > 0 ? 100.f * ip.step.update_ms  / tot : 0;
            float pD  = tot > 0 ? 100.f * ip.step.d2h_ms     / tot : 0;
            f << rp.n << "," << rp.dim << "," << ip.iter << ","
              << (ip.iter == 1 ? rp.seed_ms : 0.f) << ","
              << ip.step.assign_ms  << "," << ip.step.accum_ms
              << "," << ip.step.update_ms << "," << ip.step.d2h_ms
              << "," << tot << ","
              << pA << "," << pC << "," << pU << "," << pD << ","
              << ip.step.assign_global_read_bytes  / 1e6 << ","
              << ip.step.assign_global_write_bytes / 1e6 << ","
              << ip.step.assign_smem_bytes         / 1e6 << ","
              << ip.step.assign_ai << ","
              << ip.step.accum_global_read_bytes   / 1e6 << ","
              << ip.step.accum_global_write_bytes  / 1e6 << ","
              << ip.step.accum_smem_bytes          / 1e6 << ","
              << ip.step.accum_ai << ","
              << ip.step.update_global_read_bytes  / 1e3 << ","
              << ip.step.update_global_write_bytes / 1e3 << ","
              << ip.step.update_ai << ","
              << ip.cntA << "," << ip.cntB << ","
              << ip.centroid_dist << ","
              << (ip.converged ? 1 : 0) << ","
              << rp.total_ms << "," << rp.iters_run << "\n";
        }
    }
    printf("\nCSV written: %s\n", path.c_str());
}


// =============================================================================
// SCALING SUMMARY TABLE
// =============================================================================
static void print_scaling_table(const std::vector<RunProfile>& runs) {
    printf("\n┌──────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│  SCALING SUMMARY   dim=128   max_iters=50                                           │\n");
    printf("├────────┬───────┬────────────┬────────────┬────────────┬────────────┬────────────────┤\n");
    printf("│      n │ iters │ assign(ms) │  accum(ms) │ update(ms) │  total(ms) │  Mvecs/s       │\n");
    printf("├────────┼───────┼────────────┼────────────┼────────────┼────────────┼────────────────┤\n");

    for (auto& rp : runs) {
        float sum_assign = 0, sum_accum = 0, sum_update = 0;
        for (auto& ip : rp.iters) {
            sum_assign += ip.step.assign_ms;
            sum_accum  += ip.step.accum_ms;
            sum_update += ip.step.update_ms;
        }
        float mvecs = (rp.n / 1e6f) / (rp.total_ms / 1000.f);
        printf("│ %6d │ %5d │ %10.4f │ %10.4f │ %10.4f │ %10.3f │ %12.2f   │\n",
               rp.n, rp.iters_run,
               sum_assign, sum_accum, sum_update,
               rp.total_ms, mvecs);
    }
    printf("└────────┴───────┴────────────┴────────────┴────────────┴────────────┴────────────────┘\n");
}


// =============================================================================
// main
// =============================================================================
int main(int argc, char* argv[]) {
    printf("\n════════════════════════════════════════════════════\n");
    printf("  K-MEANS SPLIT GPU PROFILER\n");
    printf("════════════════════════════════════════════════════\n\n");

    print_gpu_info();

    int dim       = (argc >= 2) ? atoi(argv[1]) : 128;
    int max_iters = (argc >= 3) ? atoi(argv[2]) : 50;

    if (dim % 128 != 0) {
        fprintf(stderr, "ERROR: dim must be divisible by 128 (got %d)\n", dim);
        return 1;
    }

    // Cluster sizes to sweep
    const std::vector<int> sizes = {64, 128, 512, 1024, 4096, 8000};

    std::vector<RunProfile> all_runs;
    RunProfile run_8k;   // kept for detailed report

    for (int n : sizes) {
        printf("Running n=%-6d  dim=%d  max_iters=%d ...\n", n, dim, max_iters);

        auto h_vecs = make_two_blobs(n, dim);
        float* d_vecs;
        CUDA_CHECK(cudaMalloc(&d_vecs, (size_t)n * dim * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_vecs, h_vecs.data(),
                              (size_t)n * dim * sizeof(float), cudaMemcpyHostToDevice));

        // Warm-up: one dry run to heat up the GPU state machine
        {
            float *dv2;
            CUDA_CHECK(cudaMalloc(&dv2, (size_t)n * dim * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(dv2, h_vecs.data(),
                                  (size_t)n * dim * sizeof(float), cudaMemcpyHostToDevice));
            auto warm = run_profile(dv2, n, dim, 2);
            cudaFree(dv2);
        }

        auto rp = run_profile(d_vecs, n, dim, max_iters);
        all_runs.push_back(rp);

        if (n == 8000) run_8k = rp;

        cudaFree(d_vecs);
    }

    // ── Scaling summary ──────────────────────────────────────────────────────
    print_scaling_table(all_runs);

    // ── Detailed per-step breakdown for n=8000 ───────────────────────────────
    printf("\n\n──────────────────────────────────────────────────────────────────────────────────────\n");
    printf("  DETAILED PROFILE: n=8000\n");
    printf("──────────────────────────────────────────────────────────────────────────────────────\n");
    print_step_breakdown(run_8k);

    // ── CSV output ───────────────────────────────────────────────────────────
    write_csv(all_runs, "kmeans_profile_results.csv");

    printf("\nDone. Load kmeans_profile_results.csv into the Python plotter.\n\n");
    return 0;
}
