#!/bin/bash

CMD=$*

if [ -z "$CMD"];
then 
    CMD=/bin/bash
fi

home_dir=/home/tonyx/final_repo/UnofficialLaneExtraction
dataset_dir=/home/tonyx/UnofficialLaneExtraction/dataset
container_name=laneextraction
port_number=5030

docker run -d \
    -v $home_dir:/LaneExtraction\
    -v $dataset_dir:/LaneExtraction/dataset\
    --name=$container_name\
    --gpus all\
    --shm-size 32G\
    -p $port_number:6006\
    --rm -it laneextraction $CMD

docker attach laneextraction