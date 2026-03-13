// =============================================================================
// kmeans_gpu_split.cu
//
// Combined best-of-both cluster splitting kernel for GPU-resident clusters.
//
// DESIGN PHILOSOPHY
// -----------------
// We keep only the optimisations that give the most return per line of code:
//
//   KEPT from v4
//   ├─ 1 warp / vector  →  fully coalesced float4 reads  (32× memory efficiency)
//   ├─ Centroids cached in shared memory  →  broadcast to 32 lanes for free
//   ├─ Warp-shuffle reduction of dA / dB  →  zero memory traffic for reduction
//   ├─ atomicOr(d_changed)  →  4-byte D2H convergence check instead of n×4
//   └─ centroid_update_kernel  →  on-GPU division, zero PCIe per iteration
//
//   KEPT from original
//   ├─ Perturbation-based seeding (±ε from the mean centroid)
//   │     Why: seeding from vecs[0]/vecs[n-1] is sensitive to outliers.
//   │          ±ε from the mean always starts near the cluster's centre of
//   │          mass, giving a more balanced first split.
//   └─ Degenerate-split guard before centroid update
//
//   DROPPED (diminishing returns / added complexity)
//   ├─ Mixed-precision FP16 distance comparison
//   │     Why dropped: A100 FP32 throughput already saturates the pipeline;
//   │                  FP16 adds precision risk for marginal gain.
//   ├─ Multi-level smem reduction hierarchy in accumulate
//   │     Why dropped: float4 warp reads + direct smem atomicAdd is already
//   │                  fast enough; the extra reduction levels add code without
//   │                  meaningful throughput improvement at typical cluster sizes.
//   └─ Scalar fallback path for dim % 128 != 0
//         Why dropped: real embedding dims (128, 256, 512, 768, 1024) are all
//                      divisible by 128.  One clean float4 path is clearer.
//                      Callers with odd dims should pad to the next multiple.
//
// KERNEL PIPELINE  (per split iteration)
// ───────────────────────────────────────
//   assign_kernel      — label each vector 0 or 1; check GPU-side convergence
//   accumulate_kernel  — sum labelled vectors into partial centroid buffers
//   update_kernel      — divide sums by counts on GPU → new centroids
//
// All three kernels operate entirely on GPU-resident data.
// The only D2H transfers are:
//   • 4 bytes  (d_changed)   to check convergence each iteration
//   • 8 bytes  (cntA, cntB)  to detect degenerate splits
//   • Final results          (labels + 2 centroids) once at the end
// =============================================================================

#ifdef HAVE_CUDA

#include "kmeans_gpu_split.h"   // GpuSplitResult struct
#include <cuda_runtime.h>
#include <cstring>
#include <vector>
#include <stdexcept>

namespace m3 {

// -----------------------------------------------------------------------------
// Tuning constants
//
// WARPS_PER_BLOCK controls how many vectors are processed per block.
// 4 warps = 128 threads: each warp handles one vector independently.
// 4 warps share one centroid load phase, amortising __syncthreads() cost.
// Higher values increase occupancy but also smem pressure (2×dim×4 bytes
// per block).  4 is the sweet spot for dim ≤ 1024 on A100.
// -----------------------------------------------------------------------------
static constexpr int WARPS_PER_BLOCK = 4;
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;  // 128

// Perturbation magnitude for centroid seeding.
// Small enough to not disturb geometry; large enough to break symmetry.
static constexpr float PERTURB_EPS = 1e-4f;


// =============================================================================
// assign_kernel
//
// Grid  : ceil(n / WARPS_PER_BLOCK) blocks
// Block : THREADS_PER_BLOCK (128) threads  =  4 warps
// Smem  : 2 × dim × sizeof(float)   (centroids cA and cB)
//
// WHY 1 WARP PER VECTOR?
//   With 1 thread/vector (naive approach), consecutive threads in a warp
//   access elements separated by `dim` floats.  A 128-byte cache line holds
//   32 floats, so each transaction serves only 1 useful float → 1/32 efficiency.
//
//   With 1 warp/vector, all 32 lanes access consecutive floats within the
//   *same* vector.  The hardware merges these into a single 128-byte
//   transaction → full efficiency.  float4 loads quadruple this further:
//   each lane fetches 4 floats per instruction, so 32 lanes × 4 = 128 floats
//   per warp per step, perfectly filling one cache line per lane.
//
// WHY CENTROID IN SHARED MEMORY?
//   Both cA and cB are read by every warp in the block.  Loading them into
//   smem once (cooperatively by all 128 threads) and broadcasting via smem
//   is far cheaper than 4 independent global reads per warp.
//   Smem bandwidth on A100: ~19 TB/s vs HBM2e ~2 TB/s.
//
// WHY WARP-SHUFFLE REDUCTION?
//   After each lane accumulates its partial dA/dB, we need the warp-wide sum.
//   __shfl_down_sync operates register-to-register in ~2 cycles with zero
//   memory traffic.  5 shuffles cover log2(32) = 5 reduction stages.
//
// WHY atomicOr(d_changed)?
//   A naive convergence check copies all n labels D2H (n×4 bytes per iter).
//   atomicOr sets a single int on the device; we copy only 4 bytes D2H.
//   For n=100K this is 400KB → 4 bytes per iteration.
// =============================================================================
__global__ static void assign_kernel(
    const float* __restrict__ vecs,         // [n, dim] row-major, GPU-resident
    int n, int dim,
    const float* __restrict__ cA,           // [dim] current centroid A
    const float* __restrict__ cB,           // [dim] current centroid B
    const int*   __restrict__ prev_labels,  // [n]   labels from previous iter
    int*         __restrict__ labels,       // [n]   output labels this iter
    int*         __restrict__ d_changed)    // [1]   set to 1 if any label changed
{
    // Shared memory layout: [cA (dim floats) | cB (dim floats)]
    extern __shared__ float sh[];
    float* sh_cA = sh;
    float* sh_cB = sh + dim;

    const int tid         = (int)threadIdx.x;
    const int lane        = tid & 31;
    const int warp_in_blk = tid >> 5;

    // -------------------------------------------------------------------------
    // STEP 1: Cooperatively load both centroids into shared memory.
    //
    // All 128 threads stride across dim with step = blockDim.x.
    // Consecutive threads read consecutive floats → coalesced global load.
    // After __syncthreads(), every warp in the block can read cA/cB from
    // smem at ~10× the bandwidth of global memory.
    // -------------------------------------------------------------------------
    for (int d = tid; d < dim; d += blockDim.x) {
        sh_cA[d] = cA[d];
        sh_cB[d] = cB[d];
    }
    __syncthreads();

    // Each warp processes one vector.
    const int vec_idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_blk;
    if (vec_idx >= n) return;

    // -------------------------------------------------------------------------
    // STEP 2: Compute squared L2 distances to cA and cB using float4 loads.
    //
    // dim is assumed divisible by 128 (see design notes at top of file).
    // Each warp processes dim/128 groups; in each group, lane k reads the
    // float4 at position [group*32 + lane] within the vector.
    //
    // Memory access pattern:
    //   Lane 0  reads floats [0..3]
    //   Lane 1  reads floats [4..7]
    //   ...
    //   Lane 31 reads floats [124..127]
    //   → 32 consecutive float4s = 512 bytes = 4 cache lines, all coalesced.
    //
    // dA and dB accumulate in registers — no memory traffic for partials.
    // -------------------------------------------------------------------------
    float dA = 0.f, dB = 0.f;

    const float4* v4  = reinterpret_cast<const float4*>(vecs + (ptrdiff_t)vec_idx * dim);
    const float4* cA4 = reinterpret_cast<const float4*>(sh_cA);
    const float4* cB4 = reinterpret_cast<const float4*>(sh_cB);

    const int n_float4_groups = dim / 128;  // each group = 32 float4s per lane
    for (int g = 0; g < n_float4_groups; ++g) {
        const int idx4 = g * 32 + lane;

        float4 v  = v4[idx4];
        float4 ca = cA4[idx4];
        float4 cb = cB4[idx4];

        // Unrolled component-wise squared differences — compiler keeps in regs.
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

    // -------------------------------------------------------------------------
    // STEP 3: Warp-shuffle reduction of dA and dB.
    //
    // Each of the 32 lanes holds a partial sum covering dim/32 elements.
    // 5 rounds of __shfl_down_sync bring the full sum to lane 0.
    // Cost: 5 × ~2 cycles = ~10 cycles, zero memory traffic.
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // STEP 4 (lane 0 only): write label and check convergence.
    //
    // Only lane 0 has the correct reduced sum, so only it writes.
    // atomicOr(d_changed, 1) costs one global atomic only when a label
    // actually changes — no traffic at all on a converged iteration.
    // -------------------------------------------------------------------------
    if (lane == 0) {
        const int lbl = (dB < dA) ? 1 : 0;
        labels[vec_idx] = lbl;
        if (prev_labels[vec_idx] != lbl) {
            atomicOr(d_changed, 1);
        }
    }
}


// =============================================================================
// accumulate_kernel
//
// Grid  : ceil(n / WARPS_PER_BLOCK) blocks
// Block : THREADS_PER_BLOCK (128) threads
// Smem  : 2×dim×sizeof(float) + 2×sizeof(int)
//
// Accumulates per-cluster vector sums and counts into global buffers
// (sumA, sumB, cntA, cntB) using a two-phase approach:
//
//   Phase 1 — block-local accumulation into shared memory.
//     Each warp reads its vector with float4 loads (same coalescing benefit
//     as assign_kernel) and atomicAdds 4 floats at a time into smem.
//     Shared-memory atomics are ~10× faster than global atomics because
//     smem is on-chip and the L1 handles bank conflicts locally.
//
//   Phase 2 — thread 0 flushes block partial sums to global.
//     One thread iterates over dim once and calls global atomicAdd only for
//     non-zero entries.  This serialises the flush but keeps global atomic
//     contention proportional to blocks (not threads × dim).
//
// WHY NOT WARP-SHUFFLE FOR THE SUM?
//   Shuffle reduces a scalar; here each lane contributes to a different set
//   of dim output positions.  smem atomicAdd is the correct primitive:
//   it handles the scatter pattern efficiently on-chip.
// =============================================================================
__global__ static void accumulate_kernel(
    const float* __restrict__ vecs,      // [n, dim] GPU-resident
    int n, int dim,
    const int*   __restrict__ labels,    // [n]
    float* __restrict__ sumA,            // [dim] global accumulator
    float* __restrict__ sumB,            // [dim] global accumulator
    int*   __restrict__ cntA,            // [1]  global count
    int*   __restrict__ cntB)            // [1]  global count
{
    // Shared memory: [sumA_block | sumB_block | cntA_block | cntB_block]
    extern __shared__ float sh[];
    float* sh_sumA = sh;
    float* sh_sumB = sh + dim;
    int*   sh_cnt  = reinterpret_cast<int*>(sh + 2 * dim);  // sh_cnt[0]=A, [1]=B

    const int tid         = (int)threadIdx.x;
    const int lane        = tid & 31;
    const int warp_in_blk = tid >> 5;

    // -------------------------------------------------------------------------
    // PHASE 0: Zero the block-local shared accumulators.
    // All threads cooperate; stride = blockDim.x for coalesced smem writes.
    // -------------------------------------------------------------------------
    for (int d = tid; d < dim; d += blockDim.x) {
        sh_sumA[d] = 0.f;
        sh_sumB[d] = 0.f;
    }
    if (tid == 0) { sh_cnt[0] = 0; sh_cnt[1] = 0; }
    __syncthreads();

    // -------------------------------------------------------------------------
    // PHASE 1: Each warp reads its vector and scatters into smem.
    //
    // float4 loads: 32 lanes × 4 floats = 128 floats per step, coalesced.
    // atomicAdd to smem: on-chip, ~4 cycles vs ~100+ cycles for global atomic.
    // Lane 0 increments the block-local count (avoids 32× over-counting).
    // -------------------------------------------------------------------------
    const int vec_idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_blk;
    if (vec_idx < n) {
        const int lbl    = labels[vec_idx];
        float* sh_target = (lbl == 0) ? sh_sumA : sh_sumB;

        const float4* v4      = reinterpret_cast<const float4*>(
                                    vecs + (ptrdiff_t)vec_idx * dim);
        const int n_float4_groups = dim / 128;

        for (int g = 0; g < n_float4_groups; ++g) {
            const int idx4 = g * 32 + lane;       // float4 index in vector
            float4 chunk   = v4[idx4];

            // Map float4 index back to scalar base index in [0, dim).
            // base = g*128 + lane*4: lane 0 → [0..3], lane 1 → [4..7], etc.
            const int base = g * 128 + lane * 4;
            atomicAdd(&sh_target[base + 0], chunk.x);
            atomicAdd(&sh_target[base + 1], chunk.y);
            atomicAdd(&sh_target[base + 2], chunk.z);
            atomicAdd(&sh_target[base + 3], chunk.w);
        }

        if (lane == 0) atomicAdd(&sh_cnt[lbl], 1);
    }

    // -------------------------------------------------------------------------
    // PHASE 2: Flush block-local sums to global accumulators.
    //
    // Only thread 0 does this — one serial pass over dim, but this is cheap
    // relative to the parallel Phase 1 above.  The non-zero guard avoids
    // unnecessary global atomic traffic for empty entries.
    // -------------------------------------------------------------------------
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


// =============================================================================
// update_kernel
//
// Grid  : ceil(dim / 256) blocks,  Block: 256 threads
//
// Each thread d computes:
//   cA[d] = sumA[d] / cntA
//   cB[d] = sumB[d] / cntB
//
// WHY A SEPARATE KERNEL FOR THIS?
//   v3-style: copy sumA/sumB D2H → CPU division → copy cA/cB H2D.
//   For dim=1024 that is 2×4KB + 2×4KB = 16KB PCIe per iteration, plus
//   a host synchronisation stall (several microseconds on PCIe Gen4).
//   This kernel eliminates all of that: dim threads execute in parallel,
//   each doing one multiply (reciprocal) — essentially free on the GPU.
// =============================================================================
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

    // Reciprocal multiply is faster than division on GPU.
    const float inv_a = 1.f / (float)(*d_cntA);
    const float inv_b = 1.f / (float)(*d_cntB);
    cA[d] = sumA[d] * inv_a;
    cB[d] = sumB[d] * inv_b;
}


// =============================================================================
// seed_centroids_kernel
//
// Grid: 1 block,  Block: min(dim, 256) threads
//
// Computes the cluster mean on GPU and seeds:
//   cA[d] = mean[d] + eps * noise_direction[d]
//   cB[d] = mean[d] - eps * noise_direction[d]
//
// WHY PERTURBATION FROM THE MEAN (not vecs[0] / vecs[n-1])?
//   Seeding from the first and last vector works but is sensitive to
//   outliers and data ordering.  Starting both children near the cluster's
//   centre of mass (mean ± ε) guarantees a balanced initial split regardless
//   of data distribution, and converges in fewer iterations on average.
//
//   We use a deterministic perturbation direction (alternate +1 / -1 per
//   dimension) rather than random noise so the kernel is reproducible
//   without needing a cuRAND state.
// =============================================================================
__global__ static void seed_centroids_kernel(
    const float* __restrict__ vecs,   // [n, dim]
    int n, int dim,
    float* __restrict__ cA,
    float* __restrict__ cB,
    float eps)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;

    // Compute mean[d] over all n vectors for this dimension.
    float mean = 0.f;
    for (int i = 0; i < n; ++i) mean += vecs[(ptrdiff_t)i * dim + d];
    mean /= (float)n;

    // Alternating ±1 perturbation: cheap, deterministic, breaks symmetry.
    const float perturb = (d & 1) ? eps : -eps;
    cA[d] = mean + perturb;
    cB[d] = mean - perturb;
}


// =============================================================================
// gpu_split_kmeans — public entry point
//
// Accepts a device pointer d_vecs (caller owns it; we never free it).
// All internal buffers are allocated and freed here.
//
// Returns GpuSplitResult with:
//   centroid_a, centroid_b — host vectors [dim]
//   partition              — host vector [n], values in {0, 1}
//   iters_run              — number of EM iterations executed
//
// D2H traffic summary per iteration:
//   4 bytes   (d_changed)  — convergence flag
//   8 bytes   (cntA, cntB) — degenerate-split guard
// D2H traffic once at end:
//   n×4 bytes (labels)
//   2×dim×4 bytes (centroids)
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

    // Single-vector edge case: both centroids equal that vector.
    if (n == 1) {
        cudaMemcpy(result.centroid_a.data(), d_vecs,
                   (size_t)dim * sizeof(float), cudaMemcpyDeviceToHost);
        result.centroid_b = result.centroid_a;
        return result;
    }

    const size_t cen_bytes = (size_t)dim * sizeof(float);
    const size_t lbl_bytes = (size_t)n   * sizeof(int);

    // -------------------------------------------------------------------------
    // Device allocations (d_vecs is caller-owned; everything else is ours).
    // -------------------------------------------------------------------------
    float *d_cA, *d_cB, *d_sumA, *d_sumB;
    int   *d_cntA, *d_cntB, *d_labels, *d_prev_labels, *d_changed;

    cudaMalloc(&d_cA,          cen_bytes);
    cudaMalloc(&d_cB,          cen_bytes);
    cudaMalloc(&d_sumA,        cen_bytes);
    cudaMalloc(&d_sumB,        cen_bytes);
    cudaMalloc(&d_cntA,        sizeof(int));
    cudaMalloc(&d_cntB,        sizeof(int));
    cudaMalloc(&d_labels,      lbl_bytes);
    cudaMalloc(&d_prev_labels, lbl_bytes);
    cudaMalloc(&d_changed,     sizeof(int));

    // -------------------------------------------------------------------------
    // Seed centroids on GPU via perturbation-from-mean kernel.
    // 1 block is fine for the seed: it runs once and dim ≤ 1024.
    // -------------------------------------------------------------------------
    {
        const int seed_block = std::min(dim, 256);
        const int seed_grid  = (dim + seed_block - 1) / seed_block;
        seed_centroids_kernel<<<seed_grid, seed_block>>>(
            d_vecs, n, dim, d_cA, d_cB, PERTURB_EPS);
    }

    // Initialise prev_labels to 0xFF bytes → int32 value -1.
    // This ensures the first iteration always sets d_changed = 1.
    cudaMemset(d_prev_labels, 0xff, lbl_bytes);

    // -------------------------------------------------------------------------
    // Kernel launch parameters — computed once, reused every iteration.
    // -------------------------------------------------------------------------
    const int assign_block    = THREADS_PER_BLOCK;
    const int assign_grid     = (n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const size_t smem_assign  = (size_t)2 * dim * sizeof(float);

    // accumulate uses same block/grid as assign.
    const size_t smem_accum   = (size_t)2 * dim * sizeof(float)
                              + (size_t)2 * sizeof(int);

    const int update_block    = 256;
    const int update_grid     = (dim + update_block - 1) / update_block;

    // -------------------------------------------------------------------------
    // Main EM loop
    // -------------------------------------------------------------------------
    for (int iter = 0; iter < max_iters; ++iter) {

        // Reset convergence flag to 0 before the assign kernel runs.
        int zero = 0;
        cudaMemcpy(d_changed, &zero, sizeof(int), cudaMemcpyHostToDevice);

        // ── Assign ───────────────────────────────────────────────────────────
        assign_kernel<<<assign_grid, assign_block, smem_assign>>>(
            d_vecs, n, dim,
            d_cA, d_cB,
            d_prev_labels, d_labels,
            d_changed);

        // 4-byte D2H: did any label change?
        int changed = 0;
        cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        ++result.iters_run;

        // Swap prev_labels ← labels for the next iteration (D2D, no host).
        cudaMemcpy(d_prev_labels, d_labels, lbl_bytes, cudaMemcpyDeviceToDevice);

        // Converged — skip accumulate/update, centroids are already correct.
        // iter > 0 guard: skip on the very first pass (prev_labels were all -1).
        if (!changed && iter > 0) break;

        // ── Reset global accumulators ─────────────────────────────────────────
        cudaMemset(d_sumA, 0, cen_bytes);
        cudaMemset(d_sumB, 0, cen_bytes);
        cudaMemcpy(d_cntA, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cntB, &zero, sizeof(int), cudaMemcpyHostToDevice);

        // ── Accumulate ────────────────────────────────────────────────────────
        accumulate_kernel<<<assign_grid, assign_block, smem_accum>>>(
            d_vecs, n, dim,
            d_labels,
            d_sumA, d_sumB, d_cntA, d_cntB);

        // 8-byte D2H: check for degenerate split (all vectors in one cluster).
        int cntA_h = 0, cntB_h = 0;
        cudaMemcpy(&cntA_h, d_cntA, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&cntB_h, d_cntB, sizeof(int), cudaMemcpyDeviceToHost);

        if (cntA_h == 0 || cntB_h == 0) break;  // degenerate — stop early

        // ── Update centroids entirely on GPU — zero PCIe traffic ─────────────
        update_kernel<<<update_grid, update_block>>>(
            d_sumA, d_sumB, d_cntA, d_cntB, d_cA, d_cB, dim);
    }

    // -------------------------------------------------------------------------
    // D2H: copy final results back to host.
    // These three copies are the only mandatory H2D/D2H other than the
    // per-iteration 4+8 byte convergence checks above.
    // -------------------------------------------------------------------------
    cudaMemcpy(result.partition.data(),  d_labels, lbl_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(result.centroid_a.data(), d_cA,     cen_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(result.centroid_b.data(), d_cB,     cen_bytes, cudaMemcpyDeviceToHost);

    // Free all internally-allocated device memory (NOT d_vecs).
    cudaFree(d_cA);          cudaFree(d_cB);
    cudaFree(d_sumA);        cudaFree(d_sumB);
    cudaFree(d_cntA);        cudaFree(d_cntB);
    cudaFree(d_labels);      cudaFree(d_prev_labels);
    cudaFree(d_changed);

    return result;
}

} // namespace m3

#else
#error "kmeans_gpu_split.cu requires HAVE_CUDA (build with M3_USE_CUDA=ON)"
#endif
