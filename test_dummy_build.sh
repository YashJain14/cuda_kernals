nvcc -O3 -arch=sm_80 --use_fast_math -std=c++17 stage12.cu -o stage12_test -lcublas
