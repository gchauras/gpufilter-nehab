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

__global__ __launch_bounds__( WS * WS/4, MBO )
void diff(float *gauss1, float *gauss2, float* out)
{
	const int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, by = blockIdx.y;

#pragma unroll
    for (int y=0; y<4; y++) {
        int col = bx*WS+tx;
        int row = by*WS+(4*ty+y);
        int pixel = row*c_width+col;
        out[pixel] = gauss1[pixel]-gauss2[pixel];
    }
}


__host__
void algDiffGauss(
        const int& width,
        float* in_data,
        float* result,
        float& runtime)
{
    dvector<float> data (width, width);
    dvector<float> box1 (width, width);
    dvector<float> box2 (width, width);

    const int box_radius1 = 10;
    const int box_radius2 = 6;

    cpu_timer tm(0, "iP", false);

    // box filter 1
    {
        // upload buffer to GPU
        int num_iterations = 3;
        alg_setup algs;
        dvector<float> d_in, d_ybar, d_vhat, d_ysum;
        prepare_algSAT( algs, d_in, d_ybar, d_vhat, d_ysum, in_data, width, width );
        dvector<float> d_tmp_gpu( algs.width, algs.height );

        // start timer and compute 3 box filters
        tm.start();
        for (int j=0; j<num_iterations; j++) {
            if (j>0) {
                cudaMemcpy(d_in, box1, algs.width*algs.height*sizeof(float),
                        cudaMemcpyDeviceToDevice);
            }
            gpufilter::algBox(box_radius1, d_tmp_gpu, box1, d_ybar, d_vhat, d_ysum, d_in, algs );
        }
        cudaThreadSynchronize();
        tm.stop();
        runtime += tm.elapsed();

        // dont copy back to host, use it directly to compute the difference of Gaussians
        // box1.copy_to( result, algs.width, algs.height, width, width );
    }

    // box filter 2
    {
        // upload buffer to GPU
        int num_iterations = 3;
        alg_setup algs;
        dvector<float> d_in, d_ybar, d_vhat, d_ysum;
        prepare_algSAT( algs, d_in, d_ybar, d_vhat, d_ysum, in_data, width, width );
        dvector<float> d_tmp_gpu( algs.width, algs.height );

        // start timer and compute 3 box filters
        tm.start();
        for (int j=0; j<num_iterations; j++) {
            if (j>0) {
                cudaMemcpy(d_in, box2, algs.width*algs.height*sizeof(float),
                        cudaMemcpyDeviceToDevice);
            }
            gpufilter::algBox(box_radius2, d_tmp_gpu, box2, d_ybar, d_vhat, d_ysum, d_in, algs );
        }
        cudaThreadSynchronize();
        tm.stop();
        runtime += tm.elapsed();

        // dont copy back to host, use it directly to compute the difference of Gaussians
        // box2.copy_to( result, algs.width, algs.height, width, width );
    }

    // compute difference between result of box filters
    {
        dvector<float> res(width, width);

        tm.start();
        dim3 blocks(width/WS, width/WS);
        diff <<< blocks, dim3(WS, WS/4) >>>( box1, box2, res );
        cudaThreadSynchronize();
        tm.stop();
        runtime += tm.elapsed();

        res.copy_to( result, width, width, width, width );
    }
}

__global__ __launch_bounds__( WS * WS/4, MBO )
void usm(float *image, float *blur, float *out)
{
	const int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, by = blockIdx.y;

#pragma unroll
    for (int y=0; y<4; y++) {
        int col = bx*WS+tx;
        int row = by*WS+(4*ty+y);
        int pixel = row*c_width+col;
        out[pixel] = 2.0f*image[pixel] - blur[pixel];
    }
}

__host__
void unsharp_mask(float *inout,
                  const int& w,
                  const int& h,
                  const float& s,
                  float& runtime,
                  const int& extb,
                  const initcond& ic )
{
    float b10, a11, b20, a21, a22;
    weights1( s, b10, a11 );
    weights2( s, b20, a21, a22 );

    float* out    = new float[w*h];
    float* gauss1 = new float[w*h];
    float* gauss  = new float[w*h];

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

        // d_out.copy_to( inout, w * h );
        d_out.copy_to( gauss1, w * h );

        cudaFreeArray( a_in );
    }

    // alg4( inout, w, h, b20, a21, a22, extb, ic );
    {
        alg_setup algs, algs_transp;
        dvector<float> d_out, d_transp_out;
        dvector<float2> d_transp_pybar, d_transp_ezhat, d_pubar, d_evhat;
        cudaArray *a_in;

        prepare_alg4( algs, algs_transp, d_out, d_transp_out, d_transp_pybar,
                d_transp_ezhat, d_pubar, d_evhat, a_in, gauss1, w, h,
                b20, a21, a22, extb, ic );

        cpu_timer tm(0, "iP", true);
        alg4( d_out, d_transp_out, d_transp_pybar, d_transp_ezhat, d_pubar,
                d_evhat, a_in, algs, algs_transp );
        cudaThreadSynchronize();
        tm.stop();
        runtime += tm.elapsed();

        d_out.copy_to( gauss, w * h );

        cudaFreeArray( a_in );
    }

    // unsharp mask kernel
    {
        dvector<float> d_out(w, h);
        dvector<float> image(w, h);
        dvector<float> blur (w, h);
        image.copy_from( inout, w, h, w, h );
        blur .copy_from( gauss, w, h, w, h );

        cpu_timer tm(0, "iP", true);
        dim3 blocks(w/WS, h/WS);
        usm <<< blocks, dim3(WS, WS/4) >>>( image, blur, d_out );
        cudaThreadSynchronize();
        tm.stop();
        runtime += tm.elapsed();

        d_out.copy_to( out , w, h, w, h );
    }

    delete [] out;
    delete [] gauss;
    delete [] gauss1;
}

//=============================================================================
} // namespace gpufilter
//=============================================================================
