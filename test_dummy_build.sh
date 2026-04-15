nvcc -O3 -arch=sm_80 --use_fast_math -std=c++17 stage11.cu -o stage11_test -lcublas
