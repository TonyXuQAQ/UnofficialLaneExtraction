import enum
from model import LinkModel
import tensorflow as tf 
import numpy as np

import os 
import cv2
import sys 
import scipy.ndimage
import json
import math
from PIL import Image, ImageDraw
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

class InferEngine():
    def __init__(self, modelpath="./model_turningLaneExtraction_640_resnet34_poscodev3_v0seg/model110400", batchsize = 1):
        
        self.image_size = 4096
        self.window_size = 640
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.model = LinkModel(self.sess, self.window_size, batchsize=batchsize)
        self.model.restoreModel(modelpath)
        self.poscode = np.zeros((self.window_size*2, self.window_size*2,2))
        for i in range(self.window_size*2):
            self.poscode[i,:,0] = float(i) / self.window_size - 1.0
            self.poscode[:,i,1] = float(i) / self.window_size - 1.0
        self.sat_img = None
        self.direction = None
        self.skeleton = None
        self.endpoints = []
        self.data_list = os.listdir('../turningLaneValidation/testing_output')
        self.json_list = []
        
    def infer(self, sat=None, connector=None, direction = None):
        return self.model.infer(sat, connector, direction)

    def load_data(self,index):
        self.sat_img = scipy.ndimage.imread('../../dataset/sat_%s.jpg' % (index))
        self.direction = scipy.ndimage.imread('../laneAndDirectionExtraction/testing_output/skel_%s_direction.png' % (index))
        self.skeleton = scipy.ndimage.imread('../laneAndDirectionExtraction/testing_output/skel_%s.png' % (index))[:,:,0]
        self.json_list = [os.path.join('../turningLaneValidation/testing_output',x) for x in self.data_list if str(index) in x and 'json' in x]

    def get_input_crop(self,va,vb):
        connector1 = np.zeros((self.window_size,self.window_size))
        connector2 = np.zeros((self.window_size,self.window_size))
        connectorlink = np.zeros((self.window_size,self.window_size))
        connector = np.zeros((self.window_size,self.window_size,7))
        #
        center_point = [max(self.window_size//2,min(self.image_size-self.window_size//2-1,(va[0]+vb[0])//2)),max(self.window_size//2,min(self.image_size-self.window_size//2-1,(va[1]+vb[1])//2))]
        crop_img = self.sat_img[center_point[1]-self.window_size//2:center_point[1]+self.window_size//2,center_point[0]-self.window_size//2:center_point[0]+self.window_size//2].copy()
        crop_direction = self.direction[center_point[1]-self.window_size//2:center_point[1]+self.window_size//2,center_point[0]-self.window_size//2:center_point[0]+self.window_size//2].copy()
        
        x1 = va[0]-center_point[0]+self.window_size//2
        y1 = va[1]-center_point[1]+self.window_size//2
        x2 = vb[0]-center_point[0]+self.window_size//2
        y2 = vb[1]-center_point[1]+self.window_size//2
        cv2.circle(connector1, (x1,y1), 12, (255), -1)
        cv2.circle(connector2, (x2,y2), 12, (255), -1)
        connector[:,:,0] = connector1.copy() / 255.0 - 0.5
        connector[:,:,3] = connector2.copy() / 255.0 - 0.5
        connector[:,:,1:3] = self.poscode[self.window_size - y1:self.window_size * 2 - y1, self.window_size - x1:self.window_size * 2 - x1, :]
        connector[:,:,4:6] = self.poscode[self.window_size - y2:self.window_size * 2 - y2, self.window_size - x2:self.window_size * 2 - x2, :]
        connector[:,:,6] = np.copy(connectorlink) / 255.0 - 0.5

        crop_img = (crop_img.astype(np.float) / 255.0 - 0.5)
        crop_direction = (crop_direction[:,:,1:] - 127) / 127.0

        return np.expand_dims(crop_img, axis=0), np.expand_dims(connector, axis=0), np.expand_dims(crop_direction, axis=0), [x1,y1,x2,y2]
    
    def iterate_endpoints(self,tile_idx):
        with tqdm(total=len(self.json_list), unit='img') as pbar:
            for ii in range(len(self.json_list)):
                with open(self.json_list[ii],'r') as jf:
                    json_data = json.load(jf)
                if 1:#json_data['logit'][0]>0.5:
                    crop_image, connector, crop_direction, crop_info = self.get_input_crop(json_data['va'],json_data['vb'])

                    Image.fromarray(((connector[0,:,:,:3])*127+127).astype(np.uint8) ).save("testing_vis/%s__connector1.jpg"%(self.json_list[ii].split('/')[-1][:-5]))
                    Image.fromarray(((connector[0,:,:,3:6])*127+127).astype(np.uint8) ).save("testing_vis/%s_connector2.jpg"%(self.json_list[ii].split('/')[-1][:-5]))
                    vis_direction = np.zeros((self.window_size,self.window_size,3))
                    vis_direction[:,:,1:] = crop_direction[0]
                    vis_direction = vis_direction * 127.0 + 127
                    vis_direction = Image.fromarray(vis_direction.astype(np.uint8))
                    draw = ImageDraw.Draw(vis_direction)
                    draw.ellipse([crop_info[0]-5,crop_info[1]-5,crop_info[0]+5,crop_info[1]+5],fill='cyan')
                    draw.ellipse([crop_info[2]-5,crop_info[3]-5,crop_info[2]+5,crop_info[3]+5],fill='cyan')
                    vis_direction.save("testing_vis/%s_direction.jpg"%(self.json_list[ii].split('/')[-1][:-5]))
                    
                    
                    segmentation_mask = self.infer(crop_image,connector,crop_direction)[0]
                    Image.fromarray((segmentation_mask[0,:,:,0]*255).astype(np.uint8)).save("testing_vis/%s_pred.png"%(self.json_list[ii].split('/')[-1][:-5]))
                    # merge
                    center_point = json_data['center_point']
                    self.skeleton[center_point[1]-self.window_size//2:center_point[1]+self.window_size//2,\
                        center_point[0]-self.window_size//2:center_point[0]+self.window_size//2] += (segmentation_mask[0,:,:,0]*255).astype(np.uint8)
                else:
                    crop_image, connector, crop_direction, crop_info = self.get_input_crop(json_data['va'],json_data['vb'])
                    Image.fromarray(((connector[0,:,:,:3])*127+127).astype(np.uint8) ).save("testing_vis/%s__connector1_NO.jpg"%(self.json_list[ii].split('/')[-1][:-5]))
                    Image.fromarray(((connector[0,:,:,3:6])*127+127).astype(np.uint8) ).save("testing_vis/%s_connector2_NO.jpg"%(self.json_list[ii].split('/')[-1][:-5]))
                    vis_direction = np.zeros((self.window_size,self.window_size,3))
                    vis_direction[:,:,1:] = crop_direction[0]
                    vis_direction = vis_direction * 127.0 + 127
                    vis_direction = Image.fromarray(vis_direction.astype(np.uint8))
                    draw = ImageDraw.Draw(vis_direction)
                    draw.ellipse([crop_info[0]-5,crop_info[1]-5,crop_info[0]+5,crop_info[1]+5],fill='cyan')
                    draw.ellipse([crop_info[2]-5,crop_info[3]-5,crop_info[2]+5,crop_info[3]+5],fill='cyan')
                    vis_direction.save("testing_vis/%s_direction.jpg"%(self.json_list[ii].split('/')[-1][:-5]))
                pbar.update()
            Image.fromarray(self.skeleton.astype(np.uint8)).convert('RGB').save('./testing_output/%s.png'%(tile_idx))

if __name__=='__main__':
    engine = InferEngine()
    for i in range(5,6):
        engine.load_data(i)
        engine.iterate_endpoints(i)
        print('Finish NO.%s tile...'%(str(i)))