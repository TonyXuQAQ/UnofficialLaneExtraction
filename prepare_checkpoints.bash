# Get pretrained checkpoints
gdown https://drive.google.com/uc?id=1nnSxgvBzpXyt6vhir8cCDsbhVIOxy33Z
unzip models_LaneExtraction.zip

mkdir -p ./code/laneAndDirectionExtraction/model_laneExtraction_run1_640_resnet34v3_500ep/
mv models_LaneExtraction/laneAndDirectionExtraction/* ./code/laneAndDirectionExtraction/model_laneExtraction_run1_640_resnet34v3_500ep/

mkdir -p ./code/turningLaneExtraction/model_turningLaneExtraction_640_resnet34_poscodev3_v0seg/
mv models_LaneExtraction/turningLaneExtraction/* ./code/turningLaneExtraction/model_turningLaneExtraction_640_resnet34_poscodev3_v0seg/

mkdir -p ./code/turningLaneValidation/model_turningLaneValidation_run1_640_resnet34_500epseg/
mv models_LaneExtraction/turningLaneValidation/* ./code/turningLaneValidation/model_turningLaneValidation_run1_640_resnet34_500epseg/

rm -rf ./models_LaneExtraction.zip