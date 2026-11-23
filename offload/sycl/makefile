generic:
	acpp --acpp-targets=generic -O3 main.cpp -o main
blackwell:
	PATH="$(ACPP_CUDA_PATH)/bin:$$PATH" acpp --acpp-targets=cuda:sm_120 -O3 main.cpp -o main
