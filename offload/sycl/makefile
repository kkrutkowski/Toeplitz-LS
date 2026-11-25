.PHONY: generic blackwell check

generic:
	acpp --acpp-targets=generic -O3 -Wall -Wextra main.cpp -o main

blackwell:
	PATH="$(ACPP_CUDA_PATH)/bin:$$PATH" acpp --acpp-targets=cuda:sm_120 -O3 -Wall -Wextra main.cpp -o main

check: generic
	./main --list-devices
	@for dev in cuda opencl omp; do \
		if ./main --list-devices | grep -q "backend=$$dev "; then \
			if [ "$$dev" = "opencl" ] && ! command -v llvm-spirv >/dev/null 2>&1; then \
				echo "=== Skipping opencl (llvm-spirv not installed) ==="; \
				continue; \
			fi; \
			echo "=== Testing backend: $$dev ==="; \
			if [ "$$dev" = "cuda" ]; then \
				./main --device $$dev --verify-only --verify-pos-sign --prec all --min-logN 10 --max-logN 18 || exit 1; \
			else \
				./main --device $$dev --verify-only --prec all --min-logN 10 --max-logN 14 --no-autotune || exit 1; \
			fi; \
		fi; \
	done
