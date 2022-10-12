from subprocess import Popen 
import sys 

inputfile = '../dataset/sat_0.jpg'
outputfolder = './output'
model = 'resnet34v3'
# Popen("rm %s/*" % outputfolder, shell=True).wait()

# # run inference for the first stage and extract the graph
# Popen("python infer.py %s %s %s" % (inputfile, outputfolder, model), shell=True).wait()

# # turn segmentation to graph 
# Popen("python segtograph/segtograph.py %s/seg.png 64 %s/graph.p" % (outputfolder, outputfolder), shell=True).wait()

# # extract directions and create ways.json file 
# Popen("time python infer_direction.py %s %s %s" % (outputfolder + "/direction.png", outputfolder + "/graph.p", outputfolder), shell=True).wait()

# extract turning lanes
Popen("python infer_link_v4.py %s %s %s %s" % (inputfile, outputfolder + "/direction.png", outputfolder + "/graph.p", outputfolder), shell=True).wait()

# convert the graph into the format for the map editor
Popen("time python3 infer_for_editor_waylink2laneMap.py %s %s %s" % (outputfolder + "/ways.json", outputfolder + "/links.json", outputfolder + "/map_inferred.p"), shell=True).wait()
