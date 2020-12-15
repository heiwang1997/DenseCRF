#pragma once

#include "densecrf_base.h"
#include <vector>
#include <cstring>
#include <cmath>

namespace dcrf_cuda {

// CPU Implementation
template<int M>
class DenseCRFCPU : public DenseCRF {

protected:
    void expAndNormalize( float* out, const float* in, float scale = 1.0, float relax = 1.0 ) override;
    void buildMap() override;
    void stepInit() override;

public:

    // Create a dense CRF model of size N with M labels
    explicit DenseCRFCPU( int N ) : DenseCRF(N) {
        unary_ = new float[N * M];
        current_ = new float[N * M];
        next_ = new float[N * M];
        tmp_ = new float[N * M];
    }

    ~DenseCRFCPU() override {
        delete[] unary_;
        delete[] current_;
        delete[] next_;
        delete[] tmp_;
        delete[] map_;
    }

    DenseCRFCPU( DenseCRFCPU & o ) = delete;

    // Set the unary potential for all variables and labels (memory order is [x0l0 x0l1 x0l2 .. x1l0 x1l1 ...])
    void setUnaryEnergy( const float * unary ) override {
        memcpy(unary_, unary, sizeof(float) * N_ * M);
    }

    // Set the unary potential via label. Length of label array should equal to N.
    void setUnaryEnergyFromLabel(const short* labelGPU, float confidence = 0.5) override;
    void setUnaryEnergyFromLabel(const short* labelGPU, float* confidences) override;
};

static inline float very_fast_exp(float x) {
    return 1-x*(0.9999999995f-x*(0.4999999206f-x*(0.1666653019f-x*(0.0416573475f
                           -x*(0.0083013598f-x*(0.0013298820f-x*(0.0001413161f)))))));
}
static inline float fast_exp(float x) {
    bool lessZero = true;
    if (x < 0) { lessZero = false; x = -x; }
    if (x > 20) return 0;
    int mult = 0;
    while (x > 0.69*2*2*2) { mult+=3; x /= 8.0f; }
    while (x > 0.69*2*2) { mult+=2; x /= 4.0f; }
    while (x > 0.69) { mult++; x /= 2.0f; }
    x = very_fast_exp(x);
    while (mult) { mult--; x = x*x; }
    if (lessZero) { return 1 / x;
    } else { return x; }
}


template<int M>
void DenseCRFCPU<M>::expAndNormalize( float* out, const float* in, float scale /* = 1.0 */, float relax /* = 1.0 */ ) {
    auto *V = new float[ M ];
    for( int i=0; i<N_; i++ ){
        const float * b = in + i*M;
        // Find the max and subtract it so that the exp doesn't explode
        float mx = scale*b[0];
        for( int j=1; j<M; j++ )
            if( mx < scale*b[j] )
                mx = scale*b[j];
        float tt = 0;
        for( int j=0; j<M; j++ ){
            V[j] = fast_exp( scale*b[j]-mx );
            tt += V[j];
        }
        // Make it a probability
        for( int j=0; j<M; j++ )
            V[j] /= tt;

        float * a = out + i*M;
        for( int j=0; j<M; j++ )
            if (relax == 1)
                a[j] = V[j];
            else
                a[j] = (1-relax)*a[j] + relax*V[j];
    }
    delete[] V;

}

template<int M>
void DenseCRFCPU<M>::setUnaryEnergyFromLabel(const short* label, float confidence /* = 0.5 */) {
    float confidences[M];
    std::fill(confidences, confidences + M, confidence);
    setUnaryEnergyFromLabel(label, confidences);
}

template<int M>
void DenseCRFCPU<M>::setUnaryEnergyFromLabel(const short* label, float* confidences) {
    float u_energy = -log( 1.0f / M );
    float n_energies[M];
    float p_energies[M];
    for (int i = 0; i < M; ++i) {
        n_energies[i] = -log( (1.0f - confidences[i]) / (M-1) );
        p_energies[i] = -log( confidences[i] );
    }
    for (int i = 0; i < N_; ++i) {
        short tlabel = label[i];
        // Unknown.
        if (tlabel == -1) {
            for (int m = 0; m < M; ++m) {
                unary_[i * M + m] = u_energy;
            }
        } else {
            for (int m = 0; m < M; ++m) {
                unary_[i * M + m] = n_energies[tlabel];
            }
            unary_[i * M + tlabel] = p_energies[tlabel];
        }
    }
}

template<int M>
void DenseCRFCPU<M>::buildMap() {
    // Compute the maximum probability as MAP.
    if (!map_) map_ = new short[N_];
    for (int i = 0; i < N_; ++i) {
        const float* p = current_ + i * M;
        float mx = p[0];
        short imx = 0;
        for (short m = 1; m < M; ++m) {
            if (mx < p[m]) {
                mx = p[m]; imx = m;
            }
        }
        map_[i] = imx;
    }
}


template <int M>
void DenseCRFCPU<M>::stepInit() {
    for( int i=0; i<N_*M; i++ )
        next_[i] = -unary_[i];
}

}   // end namespace
