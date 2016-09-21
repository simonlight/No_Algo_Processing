
from matplotlib import pyplot as plt
from path_config import *
import json
import cv2
import os
import numpy as np
import xml.etree.cElementTree as ET
import metric_calculate
import Image
import itertools

def color_map(color):
    """Change color name to RGB list. Note that the value is not correct"""
    if color == 'b':
        return [0,0,0]
    elif color == 'g':
        return [255,0,0]
    elif color == 'r':
        return [0,255,0]
    elif color == 'c':
        return [0,0,255]
    elif color == 'm':
        return [255,255,0]
    elif color == 'y':
        return [255,0,255]
    elif color == 'k':
        return [0,255,255]
    
def visualize_fixations(fixation_path):
    for root,dirs,files in os.walk(fixation_path):
        for file in files:
            file = "2011_006217.json"
            
            
            for sy,sx in itertools.product(range(scale),repeat=2):
                filename_root = file[:-5]#.json
                fixation = json.load(open(gaze_path+file)).values()
                img = VOC2012_TRAIN_IMAGES+filename_root+'.jpg'
                img = cv2.imread(img, -1)
                rows = img.shape[0]
                cols = img.shape[1]
                out = np.zeros((rows,cols,3), dtype='uint8')
                out= np.dstack([img])
                colors = ['b','g','r','c','m','y','k']
                for cnt, subj in enumerate(fixation):
                    for fix in subj:
                        cv2.circle(out, (fix[0],fix[1]), 1, color_map(colors[cnt]), 2)
                
                xmin,ymin,xmax,ymax,cls = ground_truth_bb(VOC2012_TRAIN_ANNOTATIONS+filename_root)
#                 cv2.rectangle(out, (xmin,ymin), (xmax,ymax),(0,255,255)) 
#                 
#                 gaze_ratio_file = open(VOC2012_ACTION_ETLOSS_ACTION+cls+'/'+str(scale*scale)+'/'+filename_root+'_'+str(sy)+'_'+str(sx)+'.txt')
#                 gaze_ratio=gaze_ratio_file.readline().strip()
#                 gaze_ratio_file.close()
#                 cv2.putText(out, str(gaze_ratio[:4])+','+str(sy*6+sx),\
#                             (int(0.5*(start[0]+end[0])),int(0.5*(start[1]+end[1]))),\
#                             cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2
#                             )
                cv2.imshow(cls+' '+filename_root, out)

                k = cv2.waitKey(0)
                #space to next image
                cv2.destroyAllWindows()
                if k == 1048608:
                    break
                else:
                    continue
                
def slice_cnt(x,y,xmin, xmax, ymin, ymax):
    if x>=xmin and x<xmax and y>=ymin and y<ymax:
        return 1.0
    else:
        return 0.0

# the gaze ratio of the ground_truth bounding box
def IoU_gt_gaze_region(fixation_path):
    for root,dirs,files in os.walk(fixation_path):
        file_ratio = [0]*10
        file_num = [0]*10
        ingaze = 0
        cnt = 0

        for file in files:
            filename_root = file[:-5]#.json
            fixations = json.load(open(gaze_path+file)).values()
            
            bbs = ground_truth_bb_all_action(VOC2012_TRAIN_ANNOTATIONS+filename_root)
            for ob in fixations:
                for (point_x, point_y) in ob:
                    cnt+=1
                    for bb in bbs:
                        xmin, ymin, xmax, ymax = bb
                        if slice_cnt(int(point_x),int(point_y), xmin, xmax, ymin, ymax) ==1.0:
                            ingaze+=1.0
                            break
            
        print ingaze/cnt

#gaze ratio in the sliding window with respect to the IoU of bb&ground

def ground_truth_bb_all_action(filename):
    xmltree = ET.ElementTree(file=filename+'.xml')            
            
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

def ground_truth_bb(filerootname):
    action_names=["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]
    xmltree = ET.ElementTree(file=filerootname+'.xml')            
    for elem in xmltree.iterfind('object'):
        if len(list(elem.iter('name')))>1:
            print "error of object"
        else:
            for actions in elem.iter('actions'):
                class_index = action_names.index([action.tag for action in actions if action.text is '1'][0])
                cls = action_names[class_index]
            for coor in elem.iter('bndbox'):
                xmax = int(float([coorelem.text for coorelem in coor if coorelem.tag == 'xmax'][0]))
                xmin = int(float([coorelem.text for coorelem in coor if coorelem.tag == 'xmin'][0]))
                ymax = int(float([coorelem.text for coorelem in coor if coorelem.tag == 'ymax'][0]))
                ymin = int(float([coorelem.text for coorelem in coor if coorelem.tag == 'ymin'][0]))
    return xmin,ymin,xmax,ymax, cls

def correlation_IoU_gaze_ratio(fixation_path):
    action_names=["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]
    IoU_list=[]
    ratio_list=[]
    for root,dirs,files in os.walk(fixation_path):

        cnt = 0
        for file in files:
            cnt +=1
            print cnt
            class_index = -1
            filename_root = file[:-5]#-5 because of .json
            fixations = json.load(open(gaze_path+file)).values()
        
            xmin,ymin,xmax,ymax, cls = ground_truth_bb(VOC2012_TRAIN_ANNOTATIONS+filename_root)
            image_path = (VOC2012_TRAIN_IMAGES + filename_root)+'.jpg'
            image_res_x, image_res_y= Image.open(image_path).size 
#             print image_res_x, image_res_y
            for sy,sx in itertools.product(range(scale),repeat=2):
                hxmin, hymin = (int(sx*0.1*image_res_x), int(sy*0.1*image_res_y))   
                hxmax, hymax =(int(sx*0.1*image_res_x)+int((11-scale) * image_res_x/10),int(sy*0.1*image_res_y)+int((11-scale) * image_res_y/10))
                IoG = metric_calculate.getIoG(hxmin, hymin, hxmax, hymax, xmin, ymin, xmax, ymax)
#                 print sy,sx
                ratio_file = VOC2012_ACTION_ETLOSS_ACTION+cls+'/'+str(scale*scale)+'/'+filename_root+'_'+str(sy)+'_'+str(sx)+'.txt'
                ratio_f = open(ratio_file)
                ratio = float(ratio_f.readline().strip())
                ratio_f.close()

                if IoG>0.8 and ratio<0.1:
                    print filename_root, cls
#                 print ratio
#                 print ratio
                # any bb receives no fixations?
                # None!
#                 print IoG,ratio
                IoU_list.append(IoG)
                ratio_list.append(ratio)
        return IoU_list, ratio_list

def metric_file_analyse(metric_folder, typ):
    epsilon = '0.0'
    lbd = '1.0E-4'
    action_categories = ["jumping" ,"phoning", "playinginstrument", "reading" ,"ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]

    detection_res = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    gr_res = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    
    for category in action_categories:  
        for tradeoff in ['0.0','0.1','0.5','1.0', '10.0' ,'100.0']:
            for scale in ["50"]:
                
                detection_filename = '_'.join(["metric", typ, scale, tradeoff,str(epsilon), str(lbd), category+'_0.txt'])
                fp = os.path.join(metric_folder,detection_filename)
                f = open(fp)
                
                total_gr=0
                total_IoU_positive=0
                
                total_IoU_negative=0
                cnt=0
                tn=0
                tp=1
                for  line in f:
                    score, yp, yi, hp, image_path = line.strip().split(',')
                    grid_1, grid_2 = metric_calculate.h2GridCoor(hp, int(scale))
                    filename_root = image_path.split("/")[-1].split('.')[0]
                        
                    if yi=='1' and yp=='1':
                        tp+=1
                        cnt+=1
                        bbs = ground_truth_bb_all_action(VOC2012_TRAIN_ANNOTATIONS+filename_root)
                         
#                         ratio_file = VOC2012_ACTION_ETLOSS_ACTION+category+'/'+str(metric_calculate.convert_scale(scale))+'/'+filename_root+'_'+str(grid_1)+'_'+str(grid_2)+'.txt'
#                         ratio_f = open(ratio_file)
#                         ratio = float(ratio_f.readline().strip())
#                         ratio_f.close()
                         
#                         fixation_ratio+=ratio
                        im = Image.open(VOC2012_TRAIN_IMAGES+filename_root+'.jpg')
                        width, height = im.size
                        hxmin, hymin, hxmax, hymax = metric_calculate.h2Coor(width, height, hp, int(scale))
                        IoU = metric_calculate.getTopIoU(hxmin, hymin, hxmax, hymax, bbs)
                        total_IoU_positive += IoU
                    if yi=='-1' and yp=='-1':
                        tn += 1
                        cnt+=1
                        bbs= ground_truth_bb_all_action(VOC2012_TRAIN_ANNOTATIONS+filename_root)
                        
#                         ratio_file = VOC2012_ACTION_ETLOSS_ACTION+category+'/'+str(metric_calculate.convert_scale(scale))+'/'+filename_root+'_'+str(grid_1)+'_'+str(grid_2)+'.txt'
#                         ratio_f = open(ratio_file)
#                         ratio = float(ratio_f.readline().strip())
#                         ratio_f.close()
                        
#                         fixation_ratio+=ratio
                        im = Image.open(VOC2012_TRAIN_IMAGES+filename_root+'.jpg')
                        width, height = im.size
                        hxmin, hymin, hxmax, hymax = metric_calculate.h2Coor(width, height, hp, int(scale))
                        IoU = metric_calculate.getTopIoU(hxmin, hymin, hxmax, hymax, bbs)
                        total_IoU_negative += IoU
                print "tp:%d, scale:%s, tradeoff:%s, totalIOU:%s"%(tp, scale, tradeoff, total_IoU_positive/tp)
                print "tn:%d, scale:%s, tradeoff:%s, totalIOU:%s"%(tn, scale, tradeoff, total_IoU_negative/tn)
#                                 
#                         print "content:%s, category:%s, tradeoff:%.1f, scale:%d, epsilon:%f, lambda:%s, averageIoU:%f, TP:%f, TN:%f, FP:%f, FN:%f, acc:%f, fixation ratio:%f\n"%\
#                               (metric_folder, category, tradeoff, scale, epsilon, lambd, totalIoU/cnt, tp, tn, fp, fn, tp+tn, fixation_ratio/cnt)

def get_region(metric_filepath):
    metric = open(metric_filepath)
    return {line.strip().split(',')[-1].split('/')[-1]:line.strip().split(',')[-2] for line in metric.readlines()}

def vis_region_proposals(fixation_path, lsvm_region_dict, glsvm_region_dict):

    for root,dirs,files in os.walk(fixation_path):

        for file in files:
            class_index = -1
            filename_root = file[:-5]#-5 because of .json
            fixations = json.load(open(fixation_path+file)).values()
        
            xmin,ymin,xmax,ymax, cls = ground_truth_bb(VOC2012_TRAIN_ANNOTATIONS+filename_root)
            image_path = (VOC2012_TRAIN_IMAGES + filename_root)+'.jpg'
            image_res_x, image_res_y= Image.open(image_path).size 
            if cls != 'ridinghorse':
                continue
            
#             year='2011'
#             id='001642.ggg'
            scale = 8
    
            img = VOC2012_TRAIN_IMAGES+filename_root+'.jpg'

            img = cv2.imread(img, -1)
            rows = img.shape[0]
            cols = img.shape[1]
            out = np.zeros((rows,cols,3), dtype='uint8')
            out= np.dstack([img])
            colors = ['b','g','r','c','m','y','k']

#             for cnt, subj in enumerate(fixations):
#                 for fix in subj:
#                     cv2.circle(out, (fix[0],fix[1]), 3, [255,255,0], 2)
            for fix in fixations[0]:
                cv2.circle(out, (fix[0],fix[1]), 1, [255,255,0], 2)
            
            
            try:
                lhxmin, lhymin, lhxmax, lhymax = metric_calculate.h2Coor(cols, rows, lsvm_region_dict[filename_root],30)
                ghxmin, ghymin, ghxmax, ghymax = metric_calculate.h2Coor(cols, rows, glsvm_region_dict[filename_root],30)
                
            except KeyError:
                continue
            
            lstart = (int(lhxmin), int(lhymin))
            lend = (int(lhxmax), int(lhymax) )       
            gstart = (int(ghxmin), int(ghymin))
            gend = (int(ghxmax), int(ghymax)   )     
            cv2.rectangle(out, lstart,lend,(0,0,255), thickness=3)
#             cv2.putText(out, 'lsvm', lstart, cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,0,100),1) 
#             cv2.putText(out, 'glsvm', gstart, cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,0,100),1) 
            
            cv2.rectangle(out, gstart,gend,(255,0,0), thickness=3) 

            
            bbs = ground_truth_bb(VOC2012_TRAIN_ANNOTATIONS+filename_root)
            print bbs
            xmin,ymin,xmax,ymax, cls = bbs
            cv2.rectangle(out, (xmin,ymin), (xmax,ymax),(0,255,255), thickness=3)
            
#             if scale ==1:
#                 gaze_ratio_file = open(VOC2012_OBJECT_ETLOSS+cls+'/'+str(scale*scale)+'/'+cls+'_'+filename_root+'.txt')
#             else:
#                 gaze_ratio_file = open(VOC2012_OBJECT_ETLOSS+cls+'/'+str(scale*scale)+'/'+cls+'_'+filename_root+'_'+str(sy)+'_'+str(sx)+'.txt')
#             
# 
#             gaze_ratio=gaze_ratio_file.readline().strip()
#             gaze_ratio_file.close()
#             
#             cv2.putText(out, str(gaze_ratio[:4])+','+str(sy*scale+sx),\
#                         (int(0.5*(start[0]+end[0])),int(0.5*(start[1]+end[1]))),\
#                         cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2
#                         )
            cv2.imshow(cls+' '+filename_root, out)
            
            k = cv2.waitKey(0)
            #space to next image
            cv2.destroyAllWindows()
            if k == 1048608:
                continue
            else:
                continue
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from path_config import *
    import json
    import cv2
    import os
    import numpy as np
    import xml.etree.cElementTree as ET
    import metric_calculate
    import Image
    import itertools
    import collections
    ###CONFIG###
    fixation_path = "/local/wangxin/Data/gaze_voc_actions_stefan/train_gazes/"
    action_gaze_path="/local/wangxin/Data/gaze_voc_actions_stefan/train_gazes/"
    scale = 6
    ############
#     metric_file_analyse(metric_folder = "/local/wangxin/results/full_stefan_gaze/chain_glsvm_et/chain_glsvm_test_cv/metric/", typ="val")
#     visualize_fixations(fixation_path)
#     IoU, ratio = correlation_IoU_gaze_ratio(fixation_path)
# #     IoU=[1,2]
# #     ratio=[3,4]
#     plt.scatter(IoU,ratio,c='r', s=1)
#     plt.show()
#     IoU_gt_gaze_region(fixation_path)
    lsvm_region = get_region("/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_standard/metric_final/30/metric_train_30_0.0_1.0E-4_ridinghorse.txt")
    glsvm_region = get_region("/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_cccpgaze_positive_cv/metric_final/30/metric_train_0.2_30_0.0_1.0E-4_ridinghorse.txt")
#     visualize_fixations(VOC2012_OBJECT_EYE_PATH)
    vis_region_proposals(action_gaze_path, lsvm_region, glsvm_region)
