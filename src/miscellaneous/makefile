all:
	g++ ./pswf_remez_lut.cpp -O3 -ffast-math -o remez_test -I/usr/include/eigen3 -lquadmath

exp-coeffs:
	g++ -O2 remez_exp_lut.cpp -o remez_exp -lquadmath && ./remez_exp --emit ../../include/nanofft_exp_coeffs.h && rm -f remez_exp
