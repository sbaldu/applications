#

logs=$(ls -1 ./logs/*.log)
latest=${#logs[@]}
python vertexing.py save > ./logs/run_$((latest)).log
