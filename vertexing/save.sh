#

logs=$(ls -1 ./logs/*.log)
latest=${#logs[@]}
python vertexing.py > ./logs/run_$((latest)).log
