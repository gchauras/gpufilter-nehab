/**
 *  @file bspline_filter.cc
 *  @brief Bicubic B-Spline interpolation, simply runs example_bspline.cc for varying image widths
 *  @author Gaurav Chaurasia
 *  @date January, 2015
 */

#include <ctime>
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <iomanip>

#include <timer.h>
#include <dvector.h>

#include <cpuground.h>

#include <gpufilter.h>

#include <sat.cuh>

#define REPEATS 100


// Main
int main(int argc, char *argv[]) {
    int box_filter_radius = 5;
    int min_w = 0;
    int max_w = 0;
    int inc_w = 32;

    if (argc == 2) {
        int w = atoi(argv[1]);
        if (w%inc_w) {
            std::cerr << "Image width must be a multiple of " << inc_w << std::endl;
            return -1;
        }
        if (w) {
            min_w = w;      // run for this width only
            max_w = w;
        } else {
            min_w = 64;     // run for all widths
            max_w = 4096;
        }
    } else {
        std::cerr << "Usage: ./bspline_filter [image width], "
            << "use 0 to run all image widths" << std::endl;
        return -1;
    }

    std::cerr << "Width" << "\t" << "Bicubic_Nehab" << std::endl;

    const gpufilter::initcond ic = gpufilter::mirror;
    const int extb = 1;

    for (int in_w=min_w; in_w<=max_w; in_w+=inc_w) {
        float *in_gpu = new float[in_w*in_w];

        for (int i = 0; i < in_w*in_w; ++i)
            in_gpu[i] = rand()/float(RAND_MAX);

        gpufilter::alg_setup algs;
        gpufilter::dvector<float> d_out;
        gpufilter::dvector<float> d_transp_pybar, d_transp_ezhat, d_ptucheck, d_etvtilde;
        cudaArray *a_in;

        int w = in_w;
        int h = in_w;
        float b0 = 2.0f-std::sqrt(3.0f)+1.0f;
        float a1 = -2.0f+std::sqrt(3.0f);
        gpufilter::prepare_alg5( algs, d_out, d_transp_pybar, d_transp_ezhat, d_ptucheck, d_etvtilde, a_in, in_gpu, w, h, b0, a1, extb, ic );

        gpufilter::cpu_timer tm(in_w*in_w*REPEATS, "iP", true);
        for (int i=0; i<REPEATS; i++) {
            gpufilter::alg5( d_out, d_transp_pybar, d_transp_ezhat, d_ptucheck, d_etvtilde, a_in, algs );
        }
        cudaThreadSynchronize();
        tm.stop();
        d_out.copy_to( in_gpu, w * h );
        cudaFreeArray( a_in );

        float millisec = tm.elapsed()*1000.0f;
        float throughput = (in_w*in_w*REPEATS*1000.0f)/(millisec*1024*1024*1024);
        // std::cerr << in_w << "\t" << millisec/(REPEATS) << " ms" << std::endl;
        std::cerr << in_w << "\t" << throughput << std::endl;

        delete [] in_gpu;
    }

    return 0;
}
