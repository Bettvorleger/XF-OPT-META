#!/usr/bin/env bash
python3.11 main.py -p eil51 -m exp -i 20 -di 0.1 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log &
python3.11 main.py -p eil51 -m exp -i 20 -di 0.25 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log &
python3.11 main.py -p eil51 -m exp -i 20 -di 0.5 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log &

python3.11 main.py -p berlin52 -m exp -i 20 -di 0.1 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log &
python3.11 main.py -p berlin52 -m exp -i 20 -di 0.25 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log &
python3.11 main.py -p berlin52 -m exp -i 20 -di 0.5 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log &
wait
python3.11 main.py -p pr136 -m exp -i 20 -di 0.1 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log 
wait
python3.11 main.py -p pr136 -m exp -i 20 -di 0.25 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log 
wait
python3.11 main.py -p pr136 -m exp -i 20 -di 0.5 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log 
wait

python3.11 main.py -p pr226 -m exp -i 20 -di 0.1 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log 
wait
python3.11 main.py -p pr226 -m exp -i 20 -di 0.25 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log 
wait
python3.11 main.py -p pr226 -m exp -i 20 -di 0.5 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log 
wait

python3.11 main.py -p d198 -m exp -i 20 -di 0.1 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log
wait
python3.11 main.py -p d198 -m exp -i 20 -di 0.25 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log
wait
python3.11 main.py -p d198 -m exp -i 20 -di 0.5 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log
wait

python3.11 main.py -p rat195 -m exp -i 20 -di 0.1 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log
wait
python3.11 main.py -p rat195 -m exp -i 20 -di 0.25 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log
wait
python3.11 main.py -p rat195 -m exp -i 20 -di 0.5 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log
wait

python3.11 main.py -p gil262 -m exp -i 20 -di 0.1 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log
wait
python3.11 main.py -p gil262 -m exp -i 20 -di 0.25 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log
wait
python3.11 main.py -p gil262 -m exp -i 20 -di 0.5 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log
wait

python3.11 main.py -p lin318 -m exp -i 20 -di 0.1 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log
wait
python3.11 main.py -p lin318 -m exp -i 20 -di 0.25 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log
wait
python3.11 main.py -p lin318 -m exp -i 20 -di 0.5 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log
wait

python3.11 main.py -p pr439 -m exp -i 20 -di 0.1 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log
wait
python3.11 main.py -p pr439 -m exp -i 20 -di 0.25 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log
wait
python3.11 main.py -p pr439 -m exp -i 20 -di 0.5 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log
wait

python3.11 main.py -p fl417 -m exp -i 20 -di 0.1 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log
wait
python3.11 main.py -p fl417 -m exp -i 20 -di 0.25 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log
wait
python3.11 main.py -p fl417 -m exp -i 20 -di 0.5 -a 1 -b 5 -plb 0.075 -ppr 0.075 -ptb 0.01 -ddt 0.25 -r 'partial' &> output.log