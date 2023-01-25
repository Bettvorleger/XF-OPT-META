#!/usr/bin/env bash
python3.11 main.py -p pr136 -m opt -di 0.25 -opt gradient -oc 60 -i 6 &> output.log 
wait
python3.11 main.py -p pr136 -m opt -di 0.5 -opt gradient -oc 60 -i 6 &> output.log 
wait

python3.11 main.py -p pr226 -m opt -di 0.1 -td -opt gradient -oc 60 -i 6 &> output.log 
wait
python3.11 main.py -p pr226 -m opt -di 0.25 -td -opt gradient -oc 60 -i 6 &> output.log 
wait
python3.11 main.py -p pr226 -m opt -di 0.5 -td -opt gradient -oc 60 -i 6 &> output.log 
wait

python3.11 main.py -p pr226 -m opt -di 0.1 -td -opt gradient -oc 60 -i 6 &> output.log 
wait
python3.11 main.py -p d198 -m opt -di 0.25 -td -opt gradient -oc 60 -i 6 &> output.log
wait
python3.11 main.py -p d198 -m opt -di 0.5 -td -opt gradient -oc 60 -i 6 &> output.log