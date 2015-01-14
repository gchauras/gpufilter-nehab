/**
 *  @file diff_sat.cu
 *  @brief CUDA device code for GPU-Efficient Summed-Area Tables
 *  @author Gaurav Chaurasia
 *  @date January, 2005
 */

//== INCLUDES =================================================================

#include <symbol.h>

#include <dvector.h>

#include <gpufilter.h>
#include <gpuconsts.cuh>

#include <sat.cuh>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

__global__ __launch_bounds__( WS * WS/4, MBO )
void rgb2l(float *red, float *green, float *blue, float *lum1, float* lum2)
{
	const int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, by = blockIdx.y;

#pragma unroll
    for (int y=0; y<4; y++) {
        int col = bx*WS+tx;
        int row = by*WS+(4*ty+y);
        int pixel = row*c_width+col;
        float r = red  [pixel];
        float f = green[pixel];
        float b = blue [pixel];
        float l = r+g+b;

        lum1[pixel] = l;
        lum2[pixel] = l;
    }
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

//-- Host ---------------------------------------------------------------------

__host__
void algDiffGauss(
        const int& width,
        const int& height,
        const int& sigma_a,
        const int& sigma_b,
        float* red,
        float* blue,
        float* green,
        float* result,
        cpu_timer& tm)
{
    dvector<float> red  (width, height);
    dvector<float> blue (width, height);
    dvector<float> green(width, height);
    dvector<float> box1 (width, height);
    dvector<float> box2 (width, height);

    const int box_radius1 = 10;
    const int box_radius2 = 6;

    float* temp;

    // convert RGB to Lum
    {
        dvector<float> lum(width, height);

        tm.start();
        dim3 blocks(image_width/WS, image_height/WS, 1);
        rgb2l<<< cg_img, dim3(WS, WS/4) >>>( red, green, blue, lum );
        cudaThreadSynchronize();
        tm.stop();

        lum.copy_to( temp, width, height, width, height );
    }

    // box filter 1
    {
        int num_iterations = 3;
        alg_setup algs;
        dvector<float> d_in, d_ybar, d_vhat, d_ysum;
        prepare_algSAT( algs, d_in, d_ybar, d_vhat, d_ysum, temp, in_w, in_w );
        dvector<float> d_tmp_gpu( algs.width, algs.height );

        tm.start();
        for (int j=0; j<num_iterations; j++) {
            if (j>0) {
                cudaMemcpy(d_in, box1, algs.width*algs.height*sizeof(float),
                        cudaMemcpyDeviceToDevice);
            }
            gpufilter::algBox(box_radius1, d_tmp_gpu, box1, d_ybar, d_vhat, d_ysum, d_in, algs );
        }
        cudaThreadSynchronize();
        tm.end();

        // box1.copy_to( result, algs.width, algs.height, in_w, in_w );
    }

    // box filter 2
    {
        int num_iterations = 3;
        alg_setup algs;
        dvector<float> d_in, d_ybar, d_vhat, d_ysum;
        prepare_algSAT( algs, d_in, d_ybar, d_vhat, d_ysum, temp, in_w, in_w );
        dvector<float> d_tmp_gpu( algs.width, algs.height );

        tm.start();
        for (int j=0; j<num_iterations; j++) {
            if (j>0) {
                cudaMemcpy(d_in, box2, algs.width*algs.height*sizeof(float),
                        cudaMemcpyDeviceToDevice);
            }
            gpufilter::algBox(box_radius2, d_tmp_gpu, box2, d_ybar, d_vhat, d_ysum, d_in, algs );
        }
        cudaThreadSynchronize();
        tm.end();

        // box2.copy_to( result, algs.width, algs.height, in_w, in_w );
    }

    // compute difference between result of box filters
    {
        dvector<float> res(width, height);

        tm.start();
        dim3 blocks(image_width/WS, image_height/WS, 1);
        diff <<< cg_img, dim3(WS, WS/4) >>>( box1, box2, res );
        cudaThreadSynchronize();
        tm.stop();

        lum.copy_to( result, width, height, width, height );
    }
}
//=============================================================================
} // namespace gpufilter
//=============================================================================
