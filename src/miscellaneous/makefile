all:
	g++ ./pswf_remez_lut.cpp -O3 -ffast-math -o remez_test -I/usr/include/eigen3 -lquadmath

exp-coeffs:
	g++ -O2 remez_exp_lut.cpp -o remez_exp -lquadmath && ./remez_exp --emit ../../include/nanofft_exp_coeffs.h && rm -f remez_exp

pswf-ppoly-coeffs:
	g++ -O3 -std=c++17 pswf_ppoly_remez_lut.cpp -o pswf_ppoly_remez_lut -lquadmath && ./pswf_ppoly_remez_lut ../../offload/sycl/sycl_nufft_pswf_coeffs.hpp && rm -f pswf_ppoly_remez_lut
