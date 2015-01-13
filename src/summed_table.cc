/**
 *  @file summed_table.cc
 *  @brief Summed-Area Table, simply runs example_sat3.cc with different image widths
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
        std::cerr << "Usage: ./summed_table [image width] "
                  << "use 0 to run all image widths" << std::endl;
        return -1;
    }

    std::cerr << "Width" << "\t" << "Summed_table_Nehab" << std::endl;

    for (int in_w=min_w; in_w<=max_w; in_w+=inc_w) {
        float *in_gpu = new float[in_w*in_w];

        srand(time(0));

        for (int i = 0; i < in_w*in_w; ++i)
            in_gpu[i] = rand() % 256;

        gpufilter::alg_setup algs;
        gpufilter::dvector<float> d_in_gpu, d_ybar, d_vhat, d_ysum;
        gpufilter::prepare_algSAT( algs, d_in_gpu, d_ybar, d_vhat, d_ysum, in_gpu, in_w, in_w );
        gpufilter::dvector<float> d_out_gpu( algs.width, algs.height );

        gpufilter::cpu_timer tm(in_w*in_w*REPEATS, "iP", true);
        for (int i=0; i<REPEATS; i++) {
            gpufilter::algSAT( d_out_gpu, d_ybar, d_vhat, d_ysum, d_in_gpu, algs );
        }
        cudaThreadSynchronize();
        tm.stop();

        float millisec = tm.elapsed()*1000.0f;
        float throughput = (in_w*in_w*REPEATS*1000.0f)/(millisec*1024*1024);
        // std::cerr << in_w << "\t" << millisec/(REPEATS) << " ms" << std::endl;
        std::cerr << in_w << "\t" << throughput << std::endl;

        d_out_gpu.copy_to( in_gpu, algs.width, algs.height, in_w, in_w );

        delete [] in_gpu;
    }

    return 0;
}
