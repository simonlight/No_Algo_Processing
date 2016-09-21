
from path_config import *
from PIL import Image
from xml.dom import minidom
import collections
import os
import json
import math
import numpy as np

object_names=["dog", "cat", "motorbike", "boat", "aeroplane", "horse" ,"cow", "sofa", "diningtable", "bicycle"]
scales = [1,4,9,16,25,36,49,64]
slice = 10.0
#Extracting the fixations from the code of Ferrari eccv2014

def read_fixations(fixation_file, cls, eye_path):
    f = open(eye_path+cls + '_' + fixation_file)
    fixations = []
    for line in f: 
        x,y = line.strip().split(',')
        fixations.append([int(x),int(y)])
    f.close()
    return fixations

def slice_cnt(x,y,left, right, up, down):
    if x>left and x<=right and y>up and y<=down:
        return 1.0
    else:
        return 0.0

def calculate_gaze_ratio(eye_path=VOC2012_OBJECT_EYE_PATH, annotations = VOC2012_TRAIN_ANNOTATIONS):
    for root,dirs,files in os.walk(eye_path):
        for cnt, filename in enumerate(files):
            print cnt
            fixation_file = '_'.join(([filename.split('_')[1], filename.split('_')[2]]))
            im = fixation_file[:-4]+'.jpg'
            cls = filename.split('_')[0]
            image_res_x, image_res_y= Image.open(VOC2012_TRAIN_IMAGES +im).size
            fixations = read_fixations(fixation_file, cls, eye_path)
        
            integrate_image = np.zeros((10,10))
            for d1_inc in range(0,10):
                for d2_inc in range(0,10):
                    left = (d2_inc) * image_res_x/10.0
                    right = (d2_inc+1) * image_res_x/10.0
                    up = (d1_inc) * image_res_y/10.0
                    down = (d1_inc+1) * image_res_y/10.0
                    for point_x, point_y in fixations:
                         integrate_image[d1_inc][d2_inc]+=slice_cnt(point_x, point_y, left, right, up, down)
            for scale in scales:
                block_num = int(math.sqrt(scale))
                check=0
                for i_x in range(block_num):
                    for i_y in range(block_num):
                        ratio = np.sum(integrate_image[i_x:11-block_num+i_x, i_y:11-block_num+i_y])/len(fixations)

                        folder = VOC2012_OBJECT_ETLOSS_ACTION+cls+'/'+str(scale)+'/'
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        if scale == 1:
                            etloss_filename = folder+cls+'_' +im[:-4]+'.txt'
                        else:
                            etloss_filename = folder+cls+'_' +im[:-4]+'_'+str(i_x)+'_'+str(i_y)+'.txt'
                        loss_file = open(etloss_filename,'w')
                        loss_file.write(str(ratio))
                        loss_file.close()
#                     check+=ratio
def ground_truth_bb(filerootname, category):
    xmltree = ET.ElementTree(file="/local/wangxin/Data/VOCdevkit_trainset/VOC2012/Annotations/"+filerootname+'.xml')            
            
    #bb of objects of given class
    bbs=[]
    for elem in xmltree.iterfind('object'):
        for name in elem.iter('name'):
            
            if name.text == category:                            
                for coor in elem.iter('bndbox'):
                    xmax = [coorelem.text for coorelem in coor if coorelem.tag == 'xmax'][0]
                    xmin = [coorelem.text for coorelem in coor if coorelem.tag == 'xmin'][0]
                    ymax = [coorelem.text for coorelem in coor if coorelem.tag == 'ymax'][0]
                    ymin = [coorelem.text for coorelem in coor if coorelem.tag == 'ymin'][0]
                   
                    bbs.append([int(float(xmin)),int(float(ymin)),int(float(xmax)),int(float(ymax))])
    return bbs


def calculate_ground_truth_bb_loss(eye_path=VOC2012_OBJECT_EYE_PATH, annotations = VOC2012_TRAIN_ANNOTATIONS):
    scales= [100,90,80,70,60,50,40,30]
    for root,dirs,files in os.walk(eye_path):
        for cnt, filename in enumerate(files):
            category = filename.split('_')[0]
            filerootname = '_'.join(([filename.split('_')[1], filename.split('_')[2]]))[:-4]
            
            bbs = ground_truth_bb(filerootname, category)

            width, height= Image.open(VOC2012_TRAIN_IMAGES +filerootname+'.jpg').size
            
            for scale in scales:
                block_num = int(math.sqrt(metric_calculate.convert_scale(scale)))
                check=0
                for i_x in range(block_num):
                    for i_y in range(block_num):
                        h=i_x*block_num+i_y
                        hxmin, hymin, hxmax, hymax = metric_calculate.h2Coor(width, height, h, scale)
                        IoU = metric_calculate.getTopIoU(hxmin, hymin, hxmax, hymax, bbs)
                        
                        folder = "/local/wangxin/Data/ferrari_gaze/BBLoss/"+category+'/'+str(scale)+'/'
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        if scale == 100:
                            etloss_filename = folder+filename[:-4]+'.txt'
                        else:
                            etloss_filename = folder+filename[:-4]+'_'+str(i_x)+'_'+str(i_y)+'.txt'
                        
                        loss_file = open(etloss_filename,'w')
                        loss_file.write(str(IoU))
                        loss_file.close()
#                     check+=ratio
if __name__ == "__main__":
    import xml.etree.cElementTree as ET
    import metric_calculate
    print "ferrari bb starts"
    calculate_ground_truth_bb_loss()
    print "ferrari bb finis"
    