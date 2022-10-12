# Lane-Level Street Map Extraction from Aerial Imagery
This is the runable re-implementation code of paper Lane-Level Street Map Extraction from Aerial Imagery. This repo is **UNofficial** and is based on the [offical LaneExtraction repo](https://github.com/songtaohe/LaneExtraction), but makes it runable, especially the inference scripts.

## Notes
1. All experiments are conducted in the docker container. First go to ```./docker``` to create the image and then run ```./build_container.bash``` to run the container.

2. All experiments are conducted on a single GTX-1080Ti with python2. Since the source code of LaneExtraction is relatively old, do not use python3 or RTX3090 (RTX3090 may freaze the code due to CUDA issues). Before running the code, make sure you have selected the correct GPU in the bash scripts.

3. This repo only guarantees that it is runable and can produce reasonable results. But this repo is not well tested and the evalution part is not included. **For any bugs, open issue in the official LaneExtraction repo or this repo**.

4. For convenience, the pretrained checkpoints for inference are hardcoded written in the inference scripts. Refer to the inference scripts to change loaded checkpoints.

5. All inference outputs are stored in ```./code/laneAndDirectionExtraction/output```.

6. Find the discussion between me and the authors in issue [#9](https://github.com/songtaohe/LaneExtraction/issues/9).

## Dataset
Follow the steps of [offical LaneExtraction repo](https://github.com/songtaohe/LaneExtraction) to prepair the dataset.

### Taining
1. 
```
cd laneAndDirectionExtraction
./run_train.bash
```

2. 
```
cd turningLaneValidation
./run_train.bash
```

3. 
```
cd turningLaneExtraction
./run_train.bash
```

### Testing
1. Prepare pretrained checkpoints
```
./prepare_checkpoints.bash
```

2. 
```
cd laneAndDirectionExtraction
./run_test.bash
```

3. 
```
cd turningLaneExtraction
./run_test.bash
```
