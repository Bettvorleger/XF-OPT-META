#!/usr/bin/env bash
python3.11 main.py -p pr136 -m exp -i 20 -di 0.25 -a 2 -b 10 -plb 0.038 -ppr 0.687 -ptb 0.939 -ddt 0.390 -r 'partial' &> output.log &
python3.11 main.py -p pr136 -m exp -i 20 -di 0.5 -a 2 -b 9 -plb 0.073 -ppr 0.325 -ptb 0.667 -ddt 0.139 -r 'partial' &> output.log &
wait
python3.11 main.py -p lin318 -m exp -i 20 -di 0.25 -a 2 -b 10 -plb 0.038 -ppr 0.687 -ptb 0.939 -ddt 0.390 -r 'partial' &> output.log &
python3.11 main.py -p lin318 -m exp -i 20 -di 0.5 -a 2 -b 9 -plb 0.073 -ppr 0.325 -ptb 0.667 -ddt 0.139 -r 'partial' &> output.log &
wait
python3.11 main.py -p pr226 -m exp -i 20 -di 0.25 -a 2 -b 9 -plb 0.488 -ppr 0.471 -ptb 0.939 -ddt 0.240 -r 'partial' &> output.log &
python3.11 main.py -p pr226 -m exp -i 20 -di 0.5 -a 2 -b 9 -plb 0.428 -ppr 0.153 -ptb 0.984 -ddt 0.291 -r 'partial' &> output.log &
wait
python3.11 main.py -p pr439 -m exp -i 20 -di 0.25 -a 2 -b 9 -plb 0.488 -ppr 0.471 -ptb 0.939 -ddt 0.240 -r 'partial' &> output.log &
python3.11 main.py -p pr439 -m exp -i 20 -di 0.5 -a 2 -b 9 -plb 0.428 -ppr 0.153 -ptb 0.984 -ddt 0.291 -r 'partial' &> output.log &
wait
python3.11 main.py -p d198 -m exp -i 20 -di 0.25 -a 2 -b 10 -plb0.670 -ppr 0.090 -ptb 0.892 -ddt 0.386 -r 'full' &> output.log &
python3.11 main.py -p d198 -m exp -i 20 -di 0.5 -a 2 -b 10 -plb 0.694 -ppr 0.217 -ptb 0.666 -ddt 0.216 -r 'partial' &> output.log &
wait
python3.11 main.py -p fl417 -m exp -i 20 -di 0.25 -a 2 -b 10 -plb 0.670 -ppr 0.090 -ptb 0.892 -ddt 0.386 -r 'full' &> output.log &
python3.11 main.py -p fl417 -m exp -i 20 -di 0.5 -a 2 -b 10 -plb 0.694 -ppr 0.217 -ptb 0.666 -ddt 0.216 -r 'partial' &> output.log &