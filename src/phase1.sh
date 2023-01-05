#!/usr/bin/env bash
python3.11 main.py -p eil51 -m opt -di 0.25 -opt random -oc 30 -i 3 &>> output.log &
python3.11 main.py -p eil51 -m opt -di 0.25 -opt bayesian -oc 30 -i 3 &>> output.log &
python3.11 main.py -p eil51 -m opt -di 0.25 -opt forest -oc 30 -i 3 &>> output.log &
python3.11 main.py -p eil51 -m opt -di 0.25 -opt gradient -oc 30 -i 3 &>> output.log &

python3.11 main.py -p berlin52 -m opt -di 0.25 -opt random -oc 30 -i 3 &>> output.log &
python3.11 main.py -p berlin52 -m opt -di 0.25 -opt bayesian -oc 30 -i 3 &>> output.log &
python3.11 main.py -p berlin52 -m opt -di 0.25 -opt forest -oc 30 -i 3 &>> output.log &
python3.11 main.py -p berlin52 -m opt -di 0.25 -opt gradient -oc 30 -i 3 &>> output.log &
wait
python3.11 main.py -p pr136 -m opt -di 0.25 -opt random -oc 30 -i 3 &> output.log 
python3.11 main.py -p pr136 -m opt -di 0.25 -opt bayesian -oc 30 -i 3 &> output.log 
python3.11 main.py -p pr136 -m opt -di 0.25 -opt forest -oc 30 -i 3 &> output.log 
python3.11 main.py -p pr136 -m opt -di 0.25 -opt gradient -oc 30 -i 3 &> output.log 
wait
python3.11 main.py -p pr226 -m opt -di 0.25 -opt random -oc 30 -i 3 &> output.log 
python3.11 main.py -p pr226 -m opt -di 0.25 -opt bayesian -oc 30 -i 3 &> output.log 
python3.11 main.py -p pr226 -m opt -di 0.25 -opt forest -oc 30 -i 3 &> output.log 
python3.11 main.py -p pr226 -m opt -di 0.25 -opt gradient -oc 30 -i 3 &> output.log 
wait
python3.11 main.py -p d198 -m opt -di 0.25 -opt random -oc 30 -i 3 &> output.log 
python3.11 main.py -p d198 -m opt -di 0.25 -opt bayesian -oc 30 -i 3 &> output.log 
python3.11 main.py -p d198 -m opt -di 0.25 -opt forest -oc 30 -i 3 &> output.log 
python3.11 main.py -p d198 -m opt -di 0.25 -opt gradient -oc 30 -i 3 &> output.log 