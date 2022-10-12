from subprocess import Popen 
import sys 

for i in [0,5,6,11,12,17,18,22,25,28,31]:
    inputfile = '../../dataset/sat_%s.jpg'%i
    outputfolder = '../laneAndDirectionExtraction/output/%s'%i
    model = 'resnet34v3'
    # Popen("rm %s/*" % outputfolder, shell=True).wait()

    # extract turning lanes
    Popen("python infer_link_v4.py %s %s %s %s" % (inputfile, outputfolder + "/direction.png", outputfolder + "/graph.p", outputfolder), shell=True).wait()
    # convert the graph into the format for the map editor
    Popen("python3 infer_for_editor_waylink2laneMap.py %s %s %s" % (outputfolder + "/ways.json", outputfolder + "/links.json", outputfolder + "/map_inferred.p"), shell=True).wait()
