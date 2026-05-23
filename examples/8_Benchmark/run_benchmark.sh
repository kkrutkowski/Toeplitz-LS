#python3 ./run_benchmark.py \
#  --methods tlsf,fastchi2_103,aovdist,nifty \
#  --dataset-limit 100 \
#  --compute-limit 0.2

python3 ./run_benchmark.py \
  --methods tlsf,fastchi2_103,aovdist,nifty \
  --dataset-limit 0 \
  --compute-limit 16
