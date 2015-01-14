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

#include "cv.h"
#include "highgui.h"

#define REPEATS 100


// Main
int main(int argc, char *argv[]) {
    int box_filter_radius = 5;
    int min_w = 32;
    int max_w = 4096;
    int inc_w = 32;

    int using_image = false;

    float *in_red  = NULL;
    float *in_green= NULL;
    float *in_blue = NULL;
    float *out     = NULL;

    if (argc == 2) {
        string file_name = argv[1];
        if (using_image) {
            min_w = w;      // run for this width only
            max_w = w;
        }

    } else {
        std::cerr << "Usage: ./diff_gauss [image name], "
            << "drop argument to run all image widths" << std::endl;
        return -1;
    }

    std::cerr << "Width\tDoG_Nehab" << std::endl;

    for (int in_w=min_w; in_w<=max_w; in_w+=inc_w) {
        int width = in_w;
        int height = in_w;

        if (!using_image) {
            in_red   = new float[width*height];
            in_green = new float[width*height];
            in_blue  = new float[width*height];
            out      = new float[width*height];

            for (int i=0; i<width*height; i++) {
                in_red[i]   = rand()/float(RAND_MAX);
                in_green[i] = rand()/float(RAND_MAX);
                in_blue[i]  = rand()/float(RAND_MAX);
            }
        } else {
//            IplImage *in_img = cvLoadImage(file_in, CV_LOAD_IMAGE_UNCHANGED);
//            if( !in_img )
//                errorf("Unable to load image '%s'", file_in);
//            int in_w = in_img->width, in_h = in_img->height, depth = in_img->nChannels;
//            printf("Image is %dx%dx%d\n", in_w, in_h, depth);
//            printf("Flattening input image\n");
//            float **flat_in = new float*[depth];
//            for (int c = 0; c < depth; c++)
//                flat_in[c] = new float[in_w*in_h];
//            if( !flat_in )
//                errorf("Out of memory!");
//            for (int c = 0; c < depth; c++) {
//                cvSetImageCOI(in_img, c+1);
//                IplImage *ch_img = cvCreateImage(cvSize(in_w, in_h), in_img->depth, 1);
//                cvCopy(in_img, ch_img);
//                IplImage *uc_img = cvCreateImage(cvSize(in_w, in_h), IPL_DEPTH_8U, 1);
//                cvConvertImage(ch_img, uc_img);
//                for (int i = 0; i < in_h; ++i)
//                    for (int j = 0; j < in_w; ++j)
//                        flat_in[c][i*in_w+j] =
//                            ((uchar*)(uc_img->imageData+i*uc_img->widthStep))[j]/255.f;
//                cvReleaseImage(&ch_img);
//                cvReleaseImage(&uc_img);
            }
        }

        gpufilter::cpu_timer tm(in_w*in_w*REPEATS, "iP", true);
        for (int i=0; i<REPEATS; i++) {
            gpufilter::algDiffGauss(width, height, in_red, in_green, in_blue, out, tm);
        }

        float millisec = tm.elapsed()*1000.0f;
        float throughput = (width*width*REPEATS*1000.0f)/(millisec*1024*1024);
        std::cerr << width << "\t" << millisec/(REPEATS) << " ms" << std::endl;
        std::cerr << width << "\t" << throughput << std::endl;

        delete [] in_red;
        delete [] in_green;
        delete [] in_blue;
        delete [] out;
    }

    return 0;
}
