#!/bin/bash

bin/box_filter 0 1 2>box_filter_1.nehab.perflog
bin/box_filter 0 3 2>box_filter_3.nehab.perflog
bin/box_filter 0 6 2>box_filter_6.nehab.perflog
bin/summed_table 0 2>summed_table.nehab.perflog
bin/bspline_filter 0 0 2>bicubic_filter.nehab.perflog
bin/bspline_filter 1 0 2>biquintic_filter.nehab.perflog
bin/gaussian_filter 0 2> gaussian_filter.nehab.perflog
