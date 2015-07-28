/**
 *  @file usm.cc
 *  @brief Unsharp mask using recursive gaussian
 *  @author Gaurav Chaurasia
 *  @date April, 2015
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

#define REPEATS 100


// Main
int main(int argc, char *argv[]) {
    int min_w = 64;
    int max_w = 4096;
    int inc_w = 64;

    for (int in_w=min_w; in_w<=max_w; in_w+=inc_w) {
        int width = in_w;

        float *in_data  = new float[width*width];

        for (int i=0; i<width*width; i++) {
            in_data[i] = rand()/float(RAND_MAX);
        }

        float runtime = 0.0f;
        float sigma   = 3.0f;
        for (int i=0; i<REPEATS; i++) {
            gpufilter::unsharp_mask(in_data, in_w, in_w, sigma, runtime);
        }

        float millisec = runtime*1000.0f;
        float throughput = (width*width*REPEATS*1000.0f)/(millisec*1024*1024);
        std::cerr << width << "\t" << millisec/(REPEATS) << "\t" << throughput << std::endl;

        delete [] in_data;
    }

    return 0;
}
