from PIL import Image, ImageDraw
import numpy as np 
import pickle

import sys 
import os
sys.path.append(os.path.dirname(sys.path[0]))
from hdmapeditor.roadstructure import LaneMap

sys.path.append('../hdmapeditor')
from roadstructure import LaneMap 

for i in [0,5,6,11,12,17,18,22,25,28,31]:
    image = Image.fromarray(np.zeros((4096,4096,3)).astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(image)
    pred_graph = pickle.load(open('./output/%i/map_inferred.p'%i,'rb'))[0]
    output_graph = {}
    for n, v in pred_graph.nodes.items():
        neighbors = []
        for nei in pred_graph.neighbors[n]:
            nei = pred_graph.nodes[nei]
            draw.line([int(v[0]), int(v[1]),int(nei[0]), int(nei[1])],width=1,fill='white')
            neighbors.append(tuple(nei))
        output_graph[tuple(v)] = neighbors
    
    pickle.dump(output_graph,open('./output/%i/final_pred_graph.p'%i,'wb'),protocol=2)
    image.save('./output/%i/vis_map.png'%i)


    image = Image.fromarray(np.zeros((4096,4096,3)).astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(image)
    pred_graph = pickle.load(open('../../dataset/sat_%s_label.p'%i,'rb'))[0]
    output_graph = {}
    for n, v in pred_graph.nodes.items():
        neighbors = []
        for nei in pred_graph.neighbors[n]:
            nei = pred_graph.nodes[nei]
            draw.line([int(v[0]), int(v[1]),int(nei[0]), int(nei[1])],width=1,fill='white')
            neighbors.append(tuple(nei))
        output_graph[tuple(v)] = neighbors
    
    pickle.dump(output_graph,open('./output/%i/final_gt_graph.p'%i,'wb'),protocol=2)
    image.save('./output/%i/vis_gt_map.png'%i)

    print(i)