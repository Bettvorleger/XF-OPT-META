#!/usr/bin/env bash
python3.11 main.py -p eil51 -m exp -i 20 -di 0.1 -a 1 -b 5 -plb 0.970 -ppr 0.497 -ptb 0.942 -ddt 0.391 -r 'partial' &> output.log &
python3.11 main.py -p eil51 -m exp -i 20 -di 0.25 -a 2 -b 9 -plb 0.413 -ppr 0.239 -ptb 0.541 -ddt 0.364 -r 'full' &> output.log &
python3.11 main.py -p eil51 -m exp -i 20 -di 0.5 -a 2 -b 10 -plb 0.059 -ppr 0.969 -ptb 0.487 -ddt 0.460 -r 'full' &> output.log &

python3.11 main.py -p berlin52 -m exp -i 20 -di 0.1 -a 3 -b 9 -plb 0.045 -ppr 0.478 -ptb 0.196 -ddt 0.276 -r 'partial' &> output.log &
python3.11 main.py -p berlin52 -m exp -i 20 -di 0.25 -a 2 -b 8 -plb 0.831 -ppr 0.222 -ptb 0.313 -ddt 0.436 -r 'full' &> output.log &
python3.11 main.py -p berlin52 -m exp -i 20 -di 0.5 -a 1 -b 9 -plb 0.249 -ppr 0.774 -ptb 0.972 -ddt 0.388 -r 'partial' &> output.log &

wait
python3.11 main.py -p pr136 -m exp -i 20 -di 0.1 -a 2 -b 8 -plb 0.094 -ppr 0.008 -ptb 0.470 -ddt 0.386 -r 'partial' &> output.log
wait
python3.11 main.py -p pr136 -m exp -i 20 -di 0.25 -a 2 -b 10 -plb 0.038 -ppr 0.687 -ptb 0.939 -ddt 0.390 -r 'partial' &> output.log &
wait
python3.11 main.py -p pr136 -m exp -i 20 -di 0.5 -a 2 -b 9 -plb 0.073 -ppr 0.325 -ptb 0.667 -ddt 0.139 -r 'partial' &> output.log &

wait
python3.11 main.py -p pr226 -m exp -i 20 -di 0.1 -a 2 -b 10 -plb 0.112 -ppr 0.051 -ptb 0.436 -ddt 0.183 -r 'partial' &> output.log
wait
python3.11 main.py -p pr226 -m exp -i 20 -di 0.25 -a 2 -b 9 -plb 0.488 -ppr 0.471 -ptb 0.939 -ddt 0.240 -r 'partial' &> output.log &
wait
python3.11 main.py -p pr226 -m exp -i 20 -di 0.5 -a 2 -b 9 -plb 0.428 -ppr 0.153 -ptb 0.984 -ddt 0.291 -r 'partial' &> output.log &

wait
python3.11 main.py -p d198 -m exp -i 20 -di 0.1 -a 2 -b 9 -plb 0.002 -ppr 0.974 -ptb 0.652 -ddt 0.212 -r 'partial' &> output.log
wait
python3.11 main.py -p d198 -m exp -i 20 -di 0.25 -a 2 -b 10 -plb 0.670 -ppr 0.090 -ptb 0.892 -ddt 0.386 -r 'full' &> output.log &
wait
python3.11 main.py -p d198 -m exp -i 20 -di 0.5 -a 2 -b 10 -plb 0.694 -ppr 0.217 -ptb 0.666 -ddt 0.216 -r 'partial' &> output.log &

wait
python3.11 main.py -p rat195 -m exp -i 20 -di 0.1 -a 1 -b 5 -plb 0.970 -ppr 0.497 -ptb 0.942 -ddt 0.391 -r 'partial' &> output.log &
wait
python3.11 main.py -p rat195 -m exp -i 20 -di 0.25 -a 2 -b 9 -plb 0.413 -ppr 0.239 -ptb 0.541 -ddt 0.364 -r 'full' &> output.log &
wait
python3.11 main.py -p rat195 -m exp -i 20 -di 0.5 -a 2 -b 10 -plb 0.059 -ppr 0.969 -ptb 0.487 -ddt 0.460 -r 'full' &> output.log &

wait
python3.11 main.py -p gil262 -m exp -i 20 -di 0.1 -a 3 -b 9 -plb 0.045 -ppr 0.478 -ptb 0.196 -ddt 0.276 -r 'partial' &> output.log &
wait
python3.11 main.py -p gil262 -m exp -i 20 -di 0.25 -a 2 -b 8 -plb 0.831 -ppr 0.222 -ptb 0.313 -ddt 0.436 -r 'full' &> output.log &
wait
python3.11 main.py -p gil262 -m exp -i 20 -di 0.5 -a 1 -b 9 -plb 0.249 -ppr 0.774 -ptb 0.972 -ddt 0.388 -r 'partial' &> output.log &

wait
python3.11 main.py -p lin318 -m exp -i 20 -di 0.1 -a 2 -b 8 -plb 0.094 -ppr 0.008 -ptb 0.470 -ddt 0.386 -r 'partial' &> output.log
wait
python3.11 main.py -p lin318 -m exp -i 20 -di 0.25 -a 2 -b 10 -plb 0.038 -ppr 0.687 -ptb 0.939 -ddt 0.390 -r 'partial' &> output.log &
wait
python3.11 main.py -p lin318 -m exp -i 20 -di 0.5 -a 2 -b 9 -plb 0.073 -ppr 0.325 -ptb 0.667 -ddt 0.139 -r 'partial' &> output.log &

wait
python3.11 main.py -p pr439 -m exp -i 20 -di 0.1 -a 2 -b 10 -plb 0.112 -ppr 0.051 -ptb 0.436 -ddt 0.183 -r 'partial' &> output.log
wait
python3.11 main.py -p pr439 -m exp -i 20 -di 0.25 -a 2 -b 9 -plb 0.488 -ppr 0.471 -ptb 0.939 -ddt 0.240 -r 'partial' &> output.log &
wait
python3.11 main.py -p pr439 -m exp -i 20 -di 0.5 -a 2 -b 9 -plb 0.428 -ppr 0.153 -ptb 0.984 -ddt 0.291 -r 'partial' &> output.log &

wait
python3.11 main.py -p fl417 -m exp -i 20 -di 0.1 -a 2 -b 9 -plb 0.002 -ppr 0.974 -ptb 0.652 -ddt 0.212 -r 'partial' &> output.log
wait
python3.11 main.py -p fl417 -m exp -i 20 -di 0.25 -a 2 -b 10 -plb 0.670 -ppr 0.090 -ptb 0.892 -ddt 0.386 -r 'full' &> output.log &
wait
python3.11 main.py -p fl417 -m exp -i 20 -di 0.5 -a 2 -b 10 -plb 0.694 -ppr 0.217 -ptb 0.666 -ddt 0.216 -r 'partial' &> output.log &