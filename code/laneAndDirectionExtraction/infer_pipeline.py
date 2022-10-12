from subprocess import Popen 
import sys 


for i in [0,5,6,11,12,17,18,22,25,28,31]:
    inputfile = '../../dataset/sat_%s.jpg' % str(i)
    outputfolder = './output/%s'%str(i)
    model = 'resnet34v3'
    # Popen("rm %s/*" % outputfolder, shell=True).wait()

    Popen("mkdir -p ./output/%s"%str(i), shell=True).wait()
    # run inference for the first stage and extract the graph
    Popen("python infer.py %s %s %s" % (inputfile, outputfolder, model), shell=True).wait()
    
    # turn segmentation to graph 
    Popen("python segtograph/segtograph.py %s/seg.png 64 %s/graph.p" % (outputfolder, outputfolder), shell=True).wait()
    
    # extract directions and create ways.json file 
    Popen("python infer_direction.py %s %s %s" % (outputfolder + "/direction.png", outputfolder + "/graph.p", outputfolder), shell=True).wait()
