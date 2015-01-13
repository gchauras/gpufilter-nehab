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

    std::cerr << "Width\tBicubic_Nehab\tBiquintic_Nehab" << std::endl;

    const gpufilter::initcond ic = gpufilter::mirror;
    const int extb = 1;

    for (int in_w=min_w; in_w<=max_w; in_w+=inc_w) {
        float *in_gpu = new float[in_w*in_w];

        for (int i = 0; i < in_w*in_w; ++i)
            in_gpu[i] = rand()/float(RAND_MAX);

        float t_bicubic;
        float t_biquintic;

        // bicubic
        {
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
            // t_bicubic = millisec/(REPEATS);
            t_bicubic = (in_w*in_w*REPEATS*1000.0f)/(millisec*1024*1024);
        }

        // biquintic
        {
            gpufilter::alg_setup algs, algs_transp;
            gpufilter::dvector<float> d_out, d_transp_out;
            gpufilter::dvector<float2> d_transp_pybar, d_transp_ezhat, d_pubar, d_evhat;
            cudaArray *a_in;

            int w = in_w;
            int h = in_w;
            float b0 = 2.0f-std::sqrt(3.0f)+1.0f;
            float a1 = -2.0f+std::sqrt(3.0f);
            float a2 = 1.0f; // not actually correct, only for performance

            gpufilter::prepare_alg4( algs, algs_transp, d_out, d_transp_out, d_transp_pybar,
                               d_transp_ezhat, d_pubar, d_evhat, a_in, in_gpu, w, h,
                               b0, a1, a2, extb, ic );

            gpufilter::cpu_timer tm(in_w*in_w*REPEATS, "iP", true);
            for (int i=0; i<REPEATS; i++) {
                gpufilter::alg4( d_out, d_transp_out, d_transp_pybar, d_transp_ezhat, d_pubar,
                        d_evhat, a_in, algs, algs_transp );
            }
            cudaThreadSynchronize();
            tm.stop();

            d_out.copy_to( in_gpu, w * h );

            cudaFreeArray( a_in );

            float millisec = tm.elapsed()*1000.0f;
            // t_biquintic = millisec/(REPEATS);
            t_biquintic = (in_w*in_w*REPEATS*1000.0f)/(millisec*1024*1024);
        }

        std::cerr << in_w << "\t" << t_bicubic << "\t" << t_biquintic << std::endl;

        delete [] in_gpu;
    }

    return 0;
}
