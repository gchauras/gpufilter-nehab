/**
 *  @file box_filter.cc
 *  @brief Iterated box filtering using summed area tables for varying image widths
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
    int box_filter_radius = 10;
    int num_iterations = 0;
    int min_w = 0;
    int max_w = 0;
    int inc_w = 64;

    if (argc == 3) {
        int w = atoi(argv[1]);
        if (w%inc_w) {
            std::cerr << "Image width must be a multiple of " << inc_w << std::endl;
            return -1;
        }
        if (w) {
            min_w = w;      // run for this width only
            max_w = w;
        } else {
            min_w = inc_w;  // run for all widths
            max_w = 4096;
        }

        num_iterations = atoi(argv[2]);
        if (num_iterations<=0) {
            std::cerr << "Number of iterated box filters should be more than 0" << std::endl;
        }

    } else {
        std::cerr << "Usage: ./box_filter [image width] [filter iterations], "
            << "use 0 to run all image widths" << std::endl;
        return -1;
    }

    for (int in_w=min_w; in_w<=max_w; in_w+=inc_w) {
        float *in_gpu = new float[in_w*in_w];
        float *out_gpu= new float[in_w*in_w];

        for (int i = 0; i < in_w*in_w; ++i)
            in_gpu[i] = rand()/float(RAND_MAX);

        gpufilter::alg_setup algs;
        gpufilter::dvector<float> d_in_gpu, d_ybar, d_vhat, d_ysum;
        gpufilter::prepare_algSAT( algs, d_in_gpu, d_ybar, d_vhat, d_ysum, in_gpu, in_w, in_w );
        gpufilter::dvector<float> d_tmp_gpu( algs.width, algs.height );
        gpufilter::dvector<float> d_box    ( algs.width, algs.height );

        gpufilter::cpu_timer tm(in_w*in_w*REPEATS, "iP", true);
        for (int i=0; i<REPEATS; i++) {
            for (int j=0; j<num_iterations; j++) {
                if (j>0) {
                    cudaMemcpy(d_in_gpu, d_box, algs.width*algs.height*sizeof(float), cudaMemcpyDeviceToDevice);
                }
                gpufilter::algBox(box_filter_radius, d_tmp_gpu, d_box, d_ybar, d_vhat, d_ysum, d_in_gpu, algs );
            }
        }
        cudaThreadSynchronize();
        tm.stop();

        float millisec = tm.elapsed()*1000.0f;
        float throughput = (in_w*in_w*REPEATS*1000.0f)/(millisec*1024*1024);
        std::cerr << in_w << "\t" << millisec/(REPEATS) << "\t" << throughput << std::endl;


        d_box.copy_to( out_gpu, algs.width, algs.height, in_w, in_w );

        delete [] in_gpu;
    }

    return 0;
}
