#!/usr/bin/env bash
python3.11 main.py -p eil51 -m exp -i 20 -di 0.1 -a 1 -b 5 -plb 0.970 -ppr 0.497 -ptb 0.942 -ddt 0.391 -r 'partial' &> output.log &
python3.11 main.py -p eil51 -m exp -i 20 -di 0.25 -a 2 -b 9 -plb 0.413 -ppr 0.239 -ptb 0.541 -ddt 0.364 -r 'full' &> output.log &
python3.11 main.py -p eil51 -m exp -i 20 -di 0.5 -a 2 -b 10 -plb 0.059 -ppr 0.969 -ptb 0.487 -ddt 0.460 -r 'full' &> output.log &

python3.11 main.py -p berlin52 -m exp -i 20 -di 0.1 -a 3 -b 9 -plb 0.045 -ppr 0.478 -ptb 0.196 -ddt 0.276 -r 'partial' &> output.log &
python3.11 main.py -p berlin52 -m exp -i 20 -di 0.25 -a 2 -b 8 -plb 0.831 -ppr 0.222 -ptb 0.313 -ddt 0.436 -r 'full' &> output.log &
python3.11 main.py -p berlin52 -m exp -i 20 -di 0.5 -a 1 -b 9 -plb 0.249 -ppr 0.774 -ptb 0.972 -ddt 0.388 -r 'partial' &> output.log &

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