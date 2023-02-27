#!/usr/bin/env bash
python3.11 main.py -p pr136 -m exp -i 20 -di 0.1 -a 2 -b 8 -plb 0.094 -ppr 0.008 -ptb 0.470 -ddt 0.386 -r 'partial' &> output.log
wait
python3.11 main.py -p lin318 -m exp -i 20 -di 0.1 -a 2 -b 8 -plb 0.094 -ppr 0.008 -ptb 0.470 -ddt 0.386 -r 'partial' &> output.log
wait
python3.11 main.py -p pr226 -m exp -i 20 -di 0.1 -a 2 -b 10 -plb 0.112 -ppr 0.051 -ptb 0.436 -ddt 0.183 -r 'partial' &> output.log
wait
python3.11 main.py -p pr439 -m exp -i 20 -di 0.1 -a 2 -b 10 -plb 0.112 -ppr 0.051 -ptb 0.436 -ddt 0.183 -r 'partial' &> output.log
wait
python3.11 main.py -p d198 -m exp -i 20 -di 0.1 -a 2 -b 9 -plb 0.002 -ppr 0.974 -ptb 0.652 -ddt 0.212 -r 'partial' &> output.log
wait
python3.11 main.py -p fl417 -m exp -i 20 -di 0.1 -a 2 -b 9 -plb 0.002 -ppr 0.974 -ptb 0.652 -ddt 0.212 -r 'partial' &> output.log