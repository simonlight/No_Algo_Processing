
def valide_subjects(train_list, eye_tracking_path):
    """The subjects obtain 100% correctness are valid, otherwise, they are abandoned"""
    f = open(train_list)
    train_images = [line.strip() for line in f]
    subj_fixationcnt = collections.defaultdict(lambda:0)
    for root, dirs, files in os.walk(eye_tracking_path):
        for file in files:
            subj, year, id = file.split('_')
            filename = '_'.join([year,id])
            if subj in VOC2012_ACTION_ACTION_SUBJS and filename[:-4] in train_images:
                sub_fixationcnt[subj]+=1
    print subj_fixationcnt   

def fix_in_image(x_stimulus, y_stimulus,image_res_x, image_res_y):
    return int(x_stimulus)>0 and int(y_stimulus)>0 and int(x_stimulus)<=image_res_x and int(y_stimulus) <=image_res_y


def mapping(image_res_x, image_res_y, x_screen, y_screen, screen_res_x=1280.0, screen_res_y=1024.0):
    """Mapping function, given with Stefan's still image dataset"""
    sx=image_res_x/screen_res_x;
    sy=image_res_y/screen_res_y;
    s=max(sx,sy);
    dx=max(image_res_x*(1/sx-1/sy)/2,0);
    dy=max(image_res_y*(1/sy-1/sx)/2,0);
    x_stimulus=s*(x_screen-dx);
    y_stimulus=s*(y_screen-dy);
    return x_stimulus, y_stimulus

def valide_fixations(train_list, eye_tracking_path, valide_subjs, eye_tracking_json_path):
    """Write gazes into .json"""
    tl = open(train_list)
    train_images = [line.strip() for line in tl]
    tl.close()
    if not os.path.exists(eye_tracking_json_path):
        os.mkdir(eye_tracking_json_path)

    for root, dirs, files in os.walk(eye_tracking_path):
        for file in files:
            subj, year, id = file.split('_')
            if not subj in valide_subjs:
                continue
            filename = '_'.join([year,id])[:-8]
            if filename in train_images:
                json_path = eye_tracking_json_path+filename+'.json'
                if os.path.exists(json_path):
                    fixations_file=open(json_path,'r')
                    old_fixations = json.load(fixations_file)
                    fixations_file.close()
                else: 
                    old_fixations = collections.defaultdict(lambda:[])
                
                image_path = (VOC2012_TRAIN_IMAGES + file[4:])[:-4] 
                image_res_x, image_res_y= Image.open(image_path).size           
                
                et = open(eye_tracking_path+file,'r')
                et.readline()
                new_fixations = collections.defaultdict(lambda:[])
                
                last_time=0
                for line in et:
                    time, pupil_diameter, pupil_area, x_screen, y_screen, event = line.strip().split('\t')
                    
                    if event == 'F':
                        if int(time)-last_time>10000:
                            print line
                            print file
                        last_time=int(time)
                        x_stimulus, y_stimulus = mapping(image_res_x, image_res_y, int(float(x_screen)), int(float(y_screen)))
                        if fix_in_image(x_stimulus, y_stimulus,image_res_x, image_res_y):
                            new_fixations[str(subj).strip()].append([int(x_stimulus),int(y_stimulus)])
                et.close()
                new_fixations.update(old_fixations) 
                
#                 gaze_json = open(json_path,'w')
#                 json.dump(new_fixations,gaze_json)
#                 gaze_json.close()
                
def slice_cnt(x,y,left, right, up, down):
    if x>left and x<=right and y>up and y<=down:
        return 1.0
    else:
        return 0.0

def calculate_gaze_ratio(train_list, gaze_path):
    with open(train_list) as tl:
        train_images = [line.strip() for line in tl.readlines()]
    for c,im in enumerate(train_images):
        if c%100==0:
            print c
        fixation_file = open(os.path.join(gaze_path, im+'.json'))
                
        fixations = json.load(fixation_file)
        fixation_file.close()
        total_fixations = sum([len(observers) for observers in fixations.values()])
        
        xmldoc = minidom.parse(VOC2012_TRAIN_ANNOTATIONS + im+'.xml')
       
        image_res_x, image_res_y= Image.open(VOC2012_TRAIN_IMAGES+im+'.jpg').size
        

        integrate_image = np.zeros((10,10))
        for d1_inc in range(0,10):
            for d2_inc in range(0,10):
                left = (d2_inc) * image_res_x/10.0
                right = (d2_inc+1) * image_res_x/10.0
                up = (d1_inc) * image_res_y/10.0
                down = (d1_inc+1) * image_res_y/10.0
                for ob in fixations.values():
                    for (point_x, point_y) in ob:
                        integrate_image[d1_inc][d2_inc]+=slice_cnt(point_x, point_y, left, right, up, down)
        for scale in scales:
            block_num = int(math.sqrt(scale))
            check=0
            for i_x in range(block_num):
                for i_y in range(block_num):
                    ratio = np.sum(integrate_image[i_x:11-block_num+i_x, i_y:11-block_num+i_y])/total_fixations
                    if scale==1 and ratio!=1:
                        print im
                        print integrate_image
                        print "ratio%f"%ratio
                        
                    folder = VOC2012_ACTION_ETLOSS_ACTION+str(scale)+'/'
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    if scale == 1:
                        etratio_filename = folder + im+'.txt'
                    else:
                        print folder + im+'_'+str(i_x)+'_'+str(i_y)+'.txt'
                        etratio_filename = folder + im+'_'+str(i_x)+'_'+str(i_y)+'.txt'
#                     ratio_file = open(etratio_filename,'w')
#                     ratio_file.write(str(ratio))
#                     ratio_file.close()
                    check+=ratio


def ground_truth_bb_all_stefan(filerootname):
    xmltree = ET.ElementTree(file=filerootname+'.xml')            
            
    #bb of objects of given class
    bbs=[]
    for elem in xmltree.iterfind('object'):
        for name in elem.iter('name'):
            
            if name.text == "person":                            
                for coor in elem.iter('bndbox'):
                    xmax = [coorelem.text for coorelem in coor if coorelem.tag == 'xmax'][0]
                    xmin = [coorelem.text for coorelem in coor if coorelem.tag == 'xmin'][0]
                    ymax = [coorelem.text for coorelem in coor if coorelem.tag == 'ymax'][0]
                    ymin = [coorelem.text for coorelem in coor if coorelem.tag == 'ymin'][0]
                   
                    bbs.append([int(float(xmin)),int(float(ymin)),int(float(xmax)),int(float(ymax))])
    return bbs

def calculate_ground_truth_bb_loss(train_list, gaze_path):
    scales= [100,90,80,70,60,50,40,30]

    with open(train_list) as tl:
        train_images = [line.strip() for line in tl.readlines()]
    for c,im in enumerate(train_images):
        
        xmldoc = minidom.parse(VOC2012_TRAIN_ANNOTATIONS + im+'.xml')
       
        width, height= Image.open(VOC2012_TRAIN_IMAGES+im+'.jpg').size
        bbs = ground_truth_bb_all_stefan(VOC2012_TRAIN_ANNOTATIONS + im)

        for scale in scales:
            block_num = int(math.sqrt(metric_calculate.convert_scale(scale)))
            check=0
            for i_x in range(block_num):
                for i_y in range(block_num):
                    h=i_x*block_num+i_y
                    hxmin, hymin, hxmax, hymax = metric_calculate.h2Coor(width, height, h, scale)
                    IoU = metric_calculate.getTopIoU(hxmin, hymin, hxmax, hymax, bbs)
                    folder = "/local/wangxin/Data/full_stefan_gaze/BBLoss/"+str(scale)+'/'
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    if scale == 100:
                        etloss_filename = folder+im+'.txt'
                    else:
                        etloss_filename = folder+im+'_'+str(i_x)+'_'+str(i_y)+'.txt'
                    
                    
                    loss_file = open(etloss_filename,'w')
                    loss_file.write(str(IoU))
                    loss_file.close()

if __name__ == "__main__":
    action_names=["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]
    
    scales = [1,4,9,16,25,36,49,64]
    # scales=[1]
    #scales = [36]
    slice = 10.0
    
    from path_config import *
    from PIL import Image
    from xml.dom import minidom
    import collections
    import os
    import json
    import math
    import numpy as np
    valide_fixations("/local/wangxin/Data/VOCdevkit_trainset/VOC2012/ImageSets/Action/trainval.txt", VOC2012_ACTION_EYE_PATH, VOC2012_ACTION_VALIDE_SUBJS, VOC2012_ACTION_EYE_ACTION_JSON_PATH)
#     calculate_gaze_ratio("/local/wangxin/Data/VOCdevkit_trainset/VOC2012/ImageSets/Action/trainval.txt", VOC2012_ACTION_EYE_ACTION_JSON_PATH)
    
    #boundging box loss
#     import xml.etree.cElementTree as ET
#     import metric_calculate
#     print "stefan bb starts"
#     calculate_ground_truth_bb_loss("/local/wangxin/Data/VOCdevkit_trainset/VOC2012/ImageSets/Action/trainval.txt", VOC2012_ACTION_EYE_ACTION_JSON_PATH)
#     print "stefan bb finis"