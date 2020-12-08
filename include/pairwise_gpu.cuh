#pragma once

#include "densecrf_base.h"
#include "permutohedral_gpu.cuh"

namespace dcrf_cuda {

// Weight applying kernels for potts potential.
template<int M, int F>
__global__ static void pottsWeight(float* out, const float* in, const int n, const float pw) {
    const int ni = threadIdx.x + blockIdx.x * blockDim.x;
    const int vi = blockIdx.y;
    if (ni >= n) return;
    out[ni * M + vi] += pw * in[ni * M + vi];
}

// Initializing kernels for potts potential.
template<class T, int M, int F>
__global__ static void assembleImageFeature(int w, int h, const T* features, float posdev, float featuredev, float* out) {
    const int wi = threadIdx.x + blockIdx.x * blockDim.x;
    const int hi = threadIdx.y + blockIdx.y * blockDim.y;
    if (wi >= w || hi >= h) return;

    const int idx = hi * w + wi;
    out[idx * F + 0] = (float) wi / posdev;
    out[idx * F + 1] = (float) hi / posdev;
    #pragma unroll
    for (int i = 2; i < F; ++i) {
        out[idx * F + i] = (float) features[idx * (F - 2) + (i - 2)] / featuredev;
    }
}

template<class PT, class FT, int M, int F>
__global__ static void assembleUnorganizedFeature(int N, int pdim, const PT* positions, const FT* features, float posdev, float featuredev, float* out) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;

#pragma unroll
    for (int i = 0; i < pdim; ++i) {
        out[idx * F + i] = (float) positions[idx * pdim + i] / posdev;
    }

#pragma unroll
    for (int i = pdim; i < F; ++i) {
        out[idx * F + i] = (float) features[idx * (F - pdim) + (i - pdim)] / featuredev;
    }
}

template<int M, int F>
class PottsPotentialGPU: public PairwisePotential {
protected:
    PermutohedralLatticeGPU<float, F, M + 1>* lattice_;
    float w_;
public:
    PottsPotentialGPU(const float* features, int N, float w) : PairwisePotential(N), w_(w) {
        lattice_ = new PermutohedralLatticeGPU<float, F, M + 1>(N);
        lattice_->prepare(features);
    }
    ~PottsPotentialGPU(){
        delete lattice_;
    }
    PottsPotentialGPU( const PottsPotentialGPU&o ) = delete;

    //// Factory functions:
    // Build image-based potential: if features is NULL then applying gaussian filter only.
    template<class T = float>
    static PottsPotentialGPU<M, F>* FromImage(int w, int h, float weight, float posdev, const T* features = nullptr, float featuredev = 0.0) {
        // First assemble features:
        float* allFeatures = nullptr;
        cudaMalloc((void**)&allFeatures, sizeof(float) * F * w * h);
        dim3 blocks((w - 1) / 16 + 1, (h - 1) / 16 + 1, 1);
        dim3 blockSize(16, 16, 1);
        assembleImageFeature<T, M, F> <<<blocks, blockSize>>> (w, h, features, posdev, featuredev, allFeatures);
        cudaErrorCheck();
        auto* pt = new PottsPotentialGPU<M, F>(allFeatures, w * h, weight);
        cudaFree(allFeatures);
        return pt;
    }
    // Build linear potential:
    template<class PT = float, class FT = float>
    static PottsPotentialGPU<M, F>* FromUnorganizedData(int N, float weight, const PT* positions, float posdev, int posdim,
            const FT* features = nullptr, float featuredev = 0.0) {
        float* allFeatures = nullptr;
        cudaMalloc((void**)&allFeatures, sizeof(float) * F * N);
        dim3 blocks((N - 1) / BLOCK_SIZE + 1, 1, 1);
        dim3 blockSize(BLOCK_SIZE, 1, 1);
        assembleUnorganizedFeature<PT, FT, M, F> <<<blocks, blockSize>>> (N, posdim, positions, features, posdev, featuredev, allFeatures);
        cudaErrorCheck();
        auto* pt = new PottsPotentialGPU<M, F>(allFeatures, N, weight);
        cudaFree(allFeatures);
        return pt;
    }

    // tmp should be larger to store normalization values. (N*(M+1))
    // All pointers are device pointers
    void apply(float* out_values, const float* in_values, float* tmp) const {
        lattice_->filter(tmp, in_values);
        dim3 blocks((N_ - 1) / BLOCK_SIZE + 1, M, 1);
        dim3 blockSize(BLOCK_SIZE, 1, 1);
        pottsWeight<M, F> <<<blocks, blockSize>>> (out_values, tmp, N_, w_);
    }
};

}