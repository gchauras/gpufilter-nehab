/**
 *  @file gpufilter.cu
 *  @brief CUDA device code for GPU-Efficient Recursive Filtering Algorithms
 *  @author Diego Nehab
 *  @author Andre Maximo
 *  @date September, 2011
 */

//== INCLUDES =================================================================

#include <cmath>
#include <cstdio>
#include <cfloat>
#include <cassert>
#include <iostream>
#include <algorithm>

#include <gputex.cuh>
#include <timer.h>

#include "sat.cu"
#include "alg4.cu"
#include "alg5.cu"

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

//-- Host ---------------------------------------------------------------------

__host__
void gaussian_gpu( float **inout,
                   const int& w,
                   const int& h,
                   const int& d,
                   const float& s,
                   const int& extb,
                   const initcond& ic ) {
    float b10, a11, b20, a21, a22;
    weights1( s, b10, a11 );
    weights2( s, b20, a21, a22 );
    for (int c = 0; c < d; c++) {
        alg5( inout[c], w, h, b10, a11, extb, ic );
        alg4( inout[c], w, h, b20, a21, a22, extb, ic );
    }
}

__host__
void gaussian_gpu( float *inout,
                   const int& w,
                   const int& h,
                   const float& s,
                   const int& extb,
                   const initcond& ic ) {
    float b10, a11, b20, a21, a22;
    weights1( s, b10, a11 );
    weights2( s, b20, a21, a22 );
    alg5( inout, w, h, b10, a11, extb, ic );
    alg4( inout, w, h, b20, a21, a22, extb, ic );
}

__host__
void gaussian_gpu( float *inout,
                   const int& w,
                   const int& h,
                   const float& s,
                   float& runtime,
                   const int& extb,
                   const initcond& ic ) {
    float b10, a11, b20, a21, a22;
    weights1( s, b10, a11 );
    weights2( s, b20, a21, a22 );

    // alg5( inout, w, h, b10, a11, extb, ic );
    {

        alg_setup algs;
        dvector<float> d_out;
        dvector<float> d_transp_pybar, d_transp_ezhat, d_ptucheck, d_etvtilde;
        cudaArray *a_in;

        prepare_alg5( algs, d_out, d_transp_pybar, d_transp_ezhat, d_ptucheck,
                d_etvtilde, a_in, inout, w, h, b10, a11, extb, ic );

        cpu_timer tm(0, "iP", true);
        alg5( d_out, d_transp_pybar, d_transp_ezhat, d_ptucheck, d_etvtilde,
                a_in, algs );
        cudaThreadSynchronize();
        tm.stop();
        runtime += tm.elapsed();

        d_out.copy_to( inout, w * h );

        cudaFreeArray( a_in );
    }

    // alg4( inout, w, h, b20, a21, a22, extb, ic );
    {
        alg_setup algs, algs_transp;
        dvector<float> d_out, d_transp_out;
        dvector<float2> d_transp_pybar, d_transp_ezhat, d_pubar, d_evhat;
        cudaArray *a_in;

        prepare_alg4( algs, algs_transp, d_out, d_transp_out, d_transp_pybar,
                d_transp_ezhat, d_pubar, d_evhat, a_in, inout, w, h,
                b20, a21, a22, extb, ic );

        cpu_timer tm(0, "iP", true);
        alg4( d_out, d_transp_out, d_transp_pybar, d_transp_ezhat, d_pubar,
                d_evhat, a_in, algs, algs_transp );
        cudaThreadSynchronize();
        tm.stop();
        runtime += tm.elapsed();

        d_out.copy_to( inout, w * h );

        cudaFreeArray( a_in );
    }
}

__host__
void bspline3i_gpu( float **inout,
        const int& w,
        const int& h,
        const int& d,
        const int& extb,
        const initcond& ic ) {
    const float alpha = 2.f - sqrt(3.f);
    for (int c = 0; c < d; c++) {
        alg5( inout[c], w, h, 1.f+alpha, alpha, extb, ic );
    }
}

__host__
void bspline3i_gpu( float *inout,
        const int& w,
        const int& h,
        const int& extb,
        const initcond& ic ) {
    const float alpha = 2.f - sqrt(3.f);
    alg5( inout, w, h, 1.f+alpha, alpha, extb, ic );
}

//=============================================================================
} // namespace gpufilter
//=============================================================================
