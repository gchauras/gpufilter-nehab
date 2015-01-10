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
    int min_w = 0;
    int max_w = 0;
    int inc_w = 32;
    float sigma = 16.f;

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
        std::cerr << "Usage: ./gaussian_filter [image width], "
            << "use 0 to run all image widths" << std::endl;
        return -1;
    }

    for (int in_w=min_w; in_w<=max_w; in_w+=inc_w) {
        float *in_gpu = new float[in_w*in_w];

        for (int i = 0; i < in_w*in_w; ++i)
            in_gpu[i] = rand()/float(RAND_MAX);


        float millisec = 0.0f;
        {
            gpufilter::scoped_timer_stop sts( gpufilter::timers.gpu_add("GPU") );
            for (int i=0; i<REPEATS; i++) {
                gpufilter::gaussian_gpu( in_gpu, in_w, in_w, sigma );
            }
            millisec = sts.elapsed()*1000.0f;
        }

        std::cerr << "Width " << in_w << "\t" << millisec/(REPEATS) << " ms" << std::endl;

        delete [] in_gpu;
    }

    return 0;
}
