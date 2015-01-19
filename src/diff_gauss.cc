/**
 *  @file diff_gauss.cc
 *  @brief Difference of Gaussians computed as 3 iterated box filters
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

#define REPEATS 100


// Main
int main(int argc, char *argv[]) {
    int min_w = 64;
    int max_w = 4096;
    int inc_w = 64;

    float *in_data = NULL;
    float *out     = NULL;

    for (int in_w=min_w; in_w<=max_w; in_w+=inc_w) {
        int width = in_w;
        int height = in_w;

        in_data  = new float[width*width];
        out      = new float[width*width];

        for (int i=0; i<width*height; i++) {
            in_data[i] = rand()/float(RAND_MAX);
        }

        float runtime = 0.0f;
        for (int i=0; i<REPEATS; i++) {
            gpufilter::algDiffGauss(width, in_data, out, runtime);
        }

        float millisec = runtime*1000.0f;
        float throughput = (width*width*REPEATS*1000.0f)/(millisec*1024*1024);
        std::cerr << width << "\t" << millisec/(REPEATS) << "\t" << throughput << std::endl;

        delete [] in_data;
        delete [] out;
    }

    return 0;
}
