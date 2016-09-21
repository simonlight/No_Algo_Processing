import numpy as np
import collections
from path_config import *
import os 
import metric_calculate
import xml.etree.cElementTree as ET
import Image

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

def ground_truth_bb_all(filerootname):
    xmltree = ET.ElementTree(file=filerootname+'.xml')            
            
    #bb of objects of given class
    bbs=[]
    for category in VOC2012_OBJECT_CATEGORIES:        
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

def ground_truth_bb_all_stefan_given_action(filerootname,cls):
    xmltree = ET.ElementTree(file=filerootname+'.xml')            
            
    #bb of objects of given class
    bbs=[]
    for elem in xmltree.iterfind('object'):
        for name in elem.iter('name'):
            
            if name.text == "person":
                for actions in elem.iter('actions'):
                    for act in actions.iter(cls):
                        if act.text == "1":
                            for coor in elem.iter('bndbox'):
                                xmax = [coorelem.text for coorelem in coor if coorelem.tag == 'xmax'][0]
                                xmin = [coorelem.text for coorelem in coor if coorelem.tag == 'xmin'][0]
                                ymax = [coorelem.text for coorelem in coor if coorelem.tag == 'ymax'][0]
                                ymin = [coorelem.text for coorelem in coor if coorelem.tag == 'ymin'][0]
                               
                                bbs.append([int(float(xmin)),int(float(ymin)),int(float(xmax)),int(float(ymax))])
    return bbs

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

def ground_truth_bb(filerootname, category):
    xmltree = ET.ElementTree(file=filerootname+'.xml')            
            
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

def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1

def drawpoly(img,pts,color,thickness=1,style='dotted',):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style)

def drawrect(img,pt1,pt2,color,thickness=1,style='dotted'):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])] 
    drawpoly(img,pts,color,thickness,style)

def visualize_fixations(fixation_path):

    for root,dirs,files in os.walk(fixation_path):
        for file in files:
            cls, year, id= file.split('_')
            if cls != 'dog':
                continue
            print cls,year,id
            print file
            scale = 6
            id=id[:-4]
            filename_root= '_'.join([year,id])
    #         file = "2012_003108.json"
    #         filename_root = file[:-5]#.json
            fixation = []
            f = open(root+cls+'_'+year+'_'+id+'.txt')
            for line in f:
                x,y = line.strip().split(',')
                fixation.append([int(x),int(y)])
            f.close()
    
            img = VOC2012_TRAIN_IMAGES+year+'_'+id+'.jpg'
            img = cv2.imread(img, -1)
            rows = img.shape[0]
            cols = img.shape[1]
            print img.shape
            out = np.zeros((rows,cols,3), dtype='uint8')
            out= np.dstack([img])
#             for sy,sx in itertools.product(range(scale),repeat=2):
            for sy,sx in [[3,1],[4,2], [1,3],[4,5],[2,3]]:
#                 out = np.zeros((rows,cols,3), dtype='uint8')
#                 out= np.dstack([img])
                for x,y in fixation:
                    cv2.circle(out, (x,y),3,[255,255,0],2)
                start = (int(sx*0.1*cols), int(sy*0.1*rows))        
                end =(int(sx*0.1*cols)+int((11-scale) * cols/10),int(sy*0.1*rows)+int((11-scale) * rows/10))
#                 if sy==1 and sx==0:
#                     cv2.rectangle(out, start,end,(0,0,255), thickness=3, lineType=4) 
#                 if sy==2 and sx==3:
#                     cv2.rectangle(out, start,end,(0,255,0), thickness=3, lineType=4) 
#                 else:
#                     drawrect(out, start,end,(82,82,82), thickness=3) 

                bbs = ground_truth_bb(VOC2012_TRAIN_ANNOTATIONS+filename_root, cls)
#                 for xmin,ymin,xmax,ymax in bbs:
#                     print xmin,ymin,xmax,ymax
#                     cv2.rectangle(out, (xmin,ymin), (xmax,ymax),(0,255,255), thickness=3)
                
                if scale ==1:
                    gaze_ratio_file = open(VOC2012_OBJECT_ETLOSS+cls+'/'+str(scale*scale)+'/'+cls+'_'+filename_root+'.txt')
                else:
                    gaze_ratio_file = open(VOC2012_OBJECT_ETLOSS+cls+'/'+str(scale*scale)+'/'+cls+'_'+filename_root+'_'+str(sy)+'_'+str(sx)+'.txt')
                

                gaze_ratio=gaze_ratio_file.readline().strip()
                gaze_ratio_file.close()
                
#                 cv2.putText(out, str(gaze_ratio[:4])+','+str(sy*scale+sx),\
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

def get_region(metric_filepath):
    metric = open(metric_filepath)
    return {line.strip().split(',')[-1]:line.strip().split(',')[-2] for line in metric.readlines()}


def vis_region_proposals(fixation_path, lsvm_region_dict, glsvm_region_dict):

    for root,dirs,files in os.walk(fixation_path):
        for file in files:
            cls, year, id= file.split('_')
            if cls != 'dog':
                continue
            
#             year='2011'
#             id='001642.ggg'
            scale = 6
            id=id[:-4]
            filename_root= '_'.join([year,id])
            fixation = []
            f = open(root+cls+'_'+year+'_'+id+'.txt')
            for line in f:
                x,y = line.strip().split(',')
                fixation.append([int(x),int(y)])
            f.close()
    
            img = VOC2012_TRAIN_IMAGES+year+'_'+id+'.jpg'
            img = cv2.imread(img, -1)
            rows = img.shape[0]
            cols = img.shape[1]
            out = np.zeros((rows,cols,3), dtype='uint8')
            out= np.dstack([img])
            for x,y in fixation:
                cv2.circle(out, (x,y),3,[255,255,0],2)
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

            
            bbs = ground_truth_bb(VOC2012_TRAIN_ANNOTATIONS+filename_root, cls)
            for xmin,ymin,xmax,ymax in bbs:
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

                
def IoU_gt_gaze_region(fixations):
    object_names=["dog", "cat", "motorbike", "boat", "aeroplane", "horse" ,"cow", "sofa", "diningtable", "bicycle"]
    no_gaze=0
    for root,dirs,files in os.walk(VOC2012_OBJECT_EYE_PATH):
        file_ratio = [0]*10
        file_num = [0]*10
        for cnt, filename in enumerate(files):
            fixation_file = '_'.join(([filename.split('_')[1], filename.split('_')[2]]))
            im = fixation_file[:-4]+'.jpg'
            cls = filename.split('_')[0]
            class_index = object_names.index(cls)
            fixations = read_fixations(fixation_file, cls, VOC2012_OBJECT_EYE_PATH)
            
            bbs = ground_truth_bb(VOC2012_TRAIN_ANNOTATIONS+fixation_file[:-4], cls)
            
            ingaze = 0
            cnt=0
            for (point_x, point_y) in fixations:
                cnt+=1
                for bb in bbs:
                    if slice_cnt(int(point_x),int(point_y), obj[1], obj[0], obj[3], obj[2]) ==1.0:
                        ingaze+=1.0
                        break
            if ingaze/cnt<=0.1:
                no_gaze+=1
                print filename
                print fixations
            file_ratio[class_index] += ingaze/cnt
            file_num[class_index] += 1
        print no_gaze
        print object_names
        print file_ratio
        print file_num

def correlation_IoU_gaze_ratio(fixation_path):
    object_names=["dog", "cat", "motorbike", "boat", "aeroplane", "horse" ,"cow", "sofa", "diningtable", "bicycle"]
    IoU_list=[]
    scale=8
    ratio_list=[]
    for root,dirs,files in os.walk(fixation_path):
        file_ratio = [0]*10
        file_num = [0]*10
        for cnt, filename in enumerate(files):
            print cnt
            fixation_file = '_'.join(([filename.split('_')[1], filename.split('_')[2]]))
            filename_root = fixation_file[:-4]
            im = filename_root+'.jpg'
            cls = filename.split('_')[0]
            class_index = object_names.index(cls)
            fixations = read_fixations(fixation_file, cls, VOC2012_OBJECT_EYE_PATH)
            
            bbs = ground_truth_bb(VOC2012_TRAIN_ANNOTATIONS+fixation_file[:-4], cls)
            image_path = (VOC2012_TRAIN_IMAGES + filename_root)+'.jpg'
            image_res_x, image_res_y= Image.open(image_path).size 
#             print image_res_x, image_res_y
            
            for sy,sx in itertools.product(range(scale),repeat=2):
                hxmin, hymin = (int(sx*0.1*image_res_x), int(sy*0.1*image_res_y))   
                hxmax, hymax =(int(sx*0.1*image_res_x)+int((11-scale) * image_res_x/10),int(sy*0.1*image_res_y)+int((11-scale) * image_res_y/10))
                topIoG = metric_calculate.getTopIoG(hxmin, hymin, hxmax, hymax, bbs)
#                 print sy,sx
                ratio_file = VOC2012_OBJECT_ETLOSS+cls+'/'+str(scale*scale)+'/'+cls+'_'+filename_root+'_'+str(sy)+'_'+str(sx)+'.txt'
                ratio_f = open(ratio_file)
                ratio = float(ratio_f.readline().strip())
                ratio_f.close()

#                     if IoG>0.8 and ratio<0.1:
#                         print filename_root, cls
#                 print ratio
#                 print ratio
                # any bb receives no fixations?
                # None!
#                 print IoG,ratio
                IoU_list.append(topIoG)
                ratio_list.append(ratio)
        return IoU_list, ratio_list

def metric_file_analyse_detection_positive(metric_folder, categories):

    epsilon = '0.001'
    lbd = '1.0E-4'
    detection_types=["train", "valval", "valtest"]
    scale_cv=["50"]
    tradeoff_cv = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']
    only_tp = True
    if data_typ == "ferrari":
        etloss_root = VOC2012_OBJECT_ETLOSS
    elif data_typ == "stefan:":
        etloss_root = VOC2012_ACTION_ETLOSS
        
    detection_res = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    gr_res = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    for category in categories:
#     for category in ["dog"]:  
        for tradeoff in tradeoff_cv:
#             for scale in ["50"]:
            for scale in scale_cv:
                detection_res_3_tuple=[0]*3
                gr_res_3_tuple=[0]*3
                for typ in detection_types:
                    
                    detection_filename = '_'.join(["metric", typ, tradeoff, scale, str(epsilon), str(lbd), category+'.txt'])
                    fp = os.path.join(metric_folder, str(scale),detection_filename)
                    f = open(fp)

                    total_gr=0
                    total_IoU=0
                    
                    cnt=0
                    for  line in f:
                                                
                        yp, yi, hp, filename_root = line.strip().split(',')
                        filename_root = filename_root.split('/')[-1].strip()
                        if yi=='1' and (only_tp and yp=='1'):
                            cnt+=1
                            grid_1, grid_2 = metric_calculate.h2GridCoor(hp, int(scale))
                            gaze_file_root = filename_root
                            ratio_file = etloss_root+category+'/'+str(metric_calculate.convert_scale(int(scale)))+'/'+gaze_file_root+'_'+str(grid_1)+'_'+str(grid_2)+'.txt'
                             
                            with open(ratio_file) as ratio_f:
                                ratio = float(ratio_f.readline().strip())
                                total_gr+=ratio
                            
                            xml_filename_root = filename_root
                            bbs = ground_truth_bb_all_stefan(VOC2012_TRAIN_ANNOTATIONS+xml_filename_root)
    #                             bbs = ground_truth_bb_all(VOC2012_TRAIN_ANNOTATIONS+xml_filename_root)
                            im = Image.open(VOC2012_TRAIN_IMAGES+xml_filename_root+'.jpg')
                            width, height = im.size
                            hxmin, hymin, hxmax, hymax = metric_calculate.h2Coor(width, height, hp, int(scale))
                            IoU = metric_calculate.getTopIoU(hxmin, hymin, hxmax, hymax, bbs)
                            total_IoU += IoU
                    total_IoU /= cnt
                    total_gr /= cnt
                    detection_res_3_tuple[detection_types.index(typ)] = total_IoU
                    gr_res_3_tuple[detection_types.index(typ)] = total_gr
                detection_res[scale][tradeoff][category]= detection_res_3_tuple
                gr_res[scale][tradeoff][category]= gr_res_3_tuple
    return detection_res, gr_res

def metric_file_analyse(metric_folder, categories):
#     categories=["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]
    epsilon = '0.001'
    lbd = '1.0E-4'
    detection_types=["train", "valval", "valtest"]
    
    detection_res = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    gr_res = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    
    all_positive = True
    all_instance = True
    
    for category in categories:
#     for category in ["dog"]:  
        for tradeoff in ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']:
#             for scale in ["50"]:
            for scale in ['50']:
                detection_res_3_tuple=[0]*3
                gr_res_3_tuple=[0]*3
                for typ in detection_types:
                    
                    detection_filename = '_'.join(["metric", typ, tradeoff, scale, str(epsilon), str(lbd), category+'.txt'])
                    fp = os.path.join(metric_folder, str(scale),detection_filename)
                    f = open(fp)

                    total_gr=0
                    total_IoU=0
                    
                    cnt=0
                    for  line in f:
                                                
                        yp, yi, hp, filename_root = line.strip().split(',')
                        filename_root = filename_root.split('/')[-1].strip()
                        if True:
#                         if yi=='1':
                            cnt+=1

                            grid_1, grid_2 = metric_calculate.h2GridCoor(hp, int(scale))
                            gaze_file_root=filename_root
#                             ratio_file = VOC2012_OBJECT_ETLOSS+category+'/'+str(metric_calculate.convert_scale(int(scale)))+'/'+gaze_file_root+'_'+str(grid_1)+'_'+str(grid_2)+'.txt'
#                             ratio_file = "/local/wangxin/Data/gaze_voc_actions_stefan/ETLoss_ratio/"+str(metric_calculate.convert_scale(int(scale))) + '/'+gaze_file_root+'_'+str(grid_1)+'_'+str(grid_2)+'.txt'
#                             
#                             with open(ratio_file) as ratio_f:
#                                 ratio = float(ratio_f.readline().strip())
#                                 total_gr+=ratio
                            
                            xml_filename_root = filename_root
                            bbs = ground_truth_bb_all_stefan(VOC2012_TRAIN_ANNOTATIONS+xml_filename_root)
    #                             bbs = ground_truth_bb_all(VOC2012_TRAIN_ANNOTATIONS+xml_filename_root)
                            im = Image.open(VOC2012_TRAIN_IMAGES+xml_filename_root+'.jpg')
                            width, height = im.size
                            hxmin, hymin, hxmax, hymax = metric_calculate.h2Coor(width, height, hp, int(scale))
                            IoU = metric_calculate.getTopIoU(hxmin, hymin, hxmax, hymax, bbs)
                            total_IoU += IoU
                    total_IoU /= cnt
#                     total_gr /= cnt
                    detection_res_3_tuple[detection_types.index(typ)] = total_IoU
#                     gr_res_3_tuple[detection_types.index(typ)] = total_gr
                detection_res[scale][tradeoff][category]= detection_res_3_tuple
#                 gr_res[scale][tradeoff][category]= gr_res_3_tuple
    return detection_res, gr_res

def metric_file_analyse(metric_folder):
#     categories=["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]
    categories=["jumping"]
    epsilon = '0.0'
    lbd = '1.0E-4'
    detection_types=["train"]
#     categories = ["aeroplane" ,"cow", "dog" ,"cat" ,"motorbike" ,"boat" , "horse" , "sofa" ,"diningtable" ,"bicycle"]

    detection_res = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    gr_res = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    
    all_positive = True
    all_instance = True
    
    for scale in ['90']:
#         for tradeoff in ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']:
        for tradeoff in ['0.0']:

            total_IoU=0

            for category in categories:
#     for category in ["dog"]:  
#         for tradeoff in ['0.2']:
#             for scale in ["50"]:
                detection_res_3_tuple=[0]*3
                gr_res_3_tuple=[0]*3
                for typ in detection_types:
                    
                    detection_filename = '_'.join(["metric", typ,scale, str(tradeoff),str(epsilon), str(lbd), category+'_0.txt'])
                    fp = os.path.join(metric_folder, detection_filename)
                    f = open(fp)
                    
#                     f=open("/local/wangxin/results/full_stefan_gaze/glsvm_pos_neg/symil_cccpgaze_posneg_cv_single_split_test/metric/metric_val_50_1.0_0.0_0.0_1.0E-6_jumping.txt")
                    
                    total_gr=0
                    
                    cnt=0
                    for  line in f:
                        score, yp, yi, hp,hi, filename_root = line.strip().split(',')
#                         print score, yp, yi, hp, filename_root
                        filename_root = filename_root.split('/')[-1].strip()
#                         if True:
                        if  yi=='1':
                            cnt+=1
                            grid_1, grid_2 = metric_calculate.h2GridCoor(hp, int(scale))
                            gaze_file_root=filename_root
#                             ratio_file = VOC2012_OBJECT_ETLOSS+category+'/'+str(metric_calculate.convert_scale(int(scale)))+'/'+gaze_file_root+'_'+str(grid_1)+'_'+str(grid_2)+'.txt'
#                             ratio_file = "/local/wangxin/Data/gaze_voc_actions_stefan/ETLoss_ratio/"+str(metric_calculate.convert_scale(int(scale))) + '/'+gaze_file_root+'_'+str(grid_1)+'_'+str(grid_2)+'.txt'
#                             
#                             with open(ratio_file) as ratio_f:
#                                 ratio = float(ratio_f.readline().strip())
#                                 total_gr+=ratio
                            
                            xml_filename_root = filename_root
                            #Obj#
#                             bbs = ground_truth_bb(VOC2012_TRAIN_ANNOTATIONS+xml_filename_root,category)
#                             bbs = ground_truth_bb_all(VOC2012_TRAIN_ANNOTATIONS+xml_filename_root)
                            #action
#                             bbs = ground_truth_bb_all_stefan_given_action(VOC2012_TRAIN_ANNOTATIONS+xml_filename_root,category)
                            bbs = ground_truth_bb_all_stefan(VOC2012_TRAIN_ANNOTATIONS+xml_filename_root)
                            im = Image.open(VOC2012_TRAIN_IMAGES+xml_filename_root+'.jpg')
                            width, height = im.size
                            hxmin, hymin, hxmax, hymax = metric_calculate.h2Coor(width, height, hp, int(scale))
#                             print hxmin, hymin, hxmax, hymax
                            IoU = metric_calculate.getTopIoU(hxmin, hymin, hxmax, hymax, bbs)
#                             print filename_root, IoU, hp
                            total_IoU += IoU
                    total_IoU /= cnt
                    print category, scale, total_IoU
#                     total_gr /= cnt
                    detection_res_3_tuple[detection_types.index(typ)] = total_IoU
#                     gr_res_3_tuple[detection_types.index(typ)] = total_gr
                detection_res[scale][tradeoff][category]= detection_res_3_tuple
#                 gr_res[scale][tradeoff][category]= gr_res_3_tuple
    return detection_res, gr_res

def gr(metric_folder):
#     categories=["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]
    epsilon = '0.001'
    lbd = '1.0E-4'
    detection_types=["train"]
    categories=["jumping"]

    gr_res = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    
    
    for category in categories:
#     for category in ["dog"]:  
        for tradeoff in ['0.0']:
#             for scale in ["50"]:
            for scale in ['90']:
                for typ in detection_types:
                    
#                     detection_filename = '_'.join(["metric", typ,scale, str(tradeoff),str(epsilon), str(lbd), category+'_0.txt'])
#                     detection_filename = 'metric_val_90_0.0_0.0_1.0E-4_jumping_0.txt'
                    detection_filename = 'metric_train_90_0.0_0.0_1.0E-4_jumping_2.0_10_0.txt' #learning_rate, neuron
                    fp = os.path.join(metric_folder, detection_filename)
                    f = open(fp)

                    total_gr=0
                    
                    cnt=0
                    for  line in f:
                        
                        score, yp, yi, hp, hi, filename_root = line.strip().split(',')
                        filename_root = filename_root.split('/')[-1].strip()
                        if True:
#                         if yi=='1':
                            cnt+=1
                            print cnt
                            grid_1, grid_2 = metric_calculate.h2GridCoor(hp, int(scale))
                            gaze_file_root=filename_root
                            ratio_file = VOC2012_OBJECT_ETLOSS+category+'/'+str(metric_calculate.convert_scale(int(scale)))+'/'+gaze_file_root+'_'+str(grid_1)+'_'+str(grid_2)+'.txt'
                            ratio_file = "/local/wangxin/Data/gaze_voc_actions_stefan/ETLoss_ratio/"+str(metric_calculate.convert_scale(int(scale))) + '/'+gaze_file_root+'_'+str(grid_1)+'_'+str(grid_2)+'.txt'
#                             
                            with open(ratio_file) as ratio_f:
                                ratio = float(ratio_f.readline().strip())
                                total_gr+=ratio
                    total_gr /= cnt
                print total_gr
    return gr_res

def iou_hi_hp(metric_folder):
    categories=["jumping"]
    epsilon = '0.0'
    lbd = '1.0E-4'
    detection_types=["train"]
#     categories = ["aeroplane" ,"cow", "dog" ,"cat" ,"motorbike" ,"boat" , "horse" , "sofa" ,"diningtable" ,"bicycle"]

    detection_res = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    gr_res = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    
    all_positive = True
    all_instance = True
    
    for scale in ['90']:
#         for tradeoff in ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']:
        for tradeoff in ['0.0']:

            total_IoU=0

            for category in categories:
                detection_res_3_tuple=[0]*3
                gr_res_3_tuple=[0]*3
                for typ in detection_types:
                    detection_filename = '_'.join(["metric", typ,scale, str(tradeoff),str(epsilon), str(lbd), category+'_0.txt'])
#                     detection_filename = 'metric_train_90_0.0_0.0_1.0E-4_jumping_5.0_10_0.txt'
                    fp = os.path.join(metric_folder, detection_filename)
                    f = open(fp)
                    
                    total_gr=0
                    
                    cnt=0
                    for  line in f:
                        cnt+=1
                        score, yp, yi, hp,hi, filename_root = line.strip().split(',')
                        width, height = 1,1
                        phxmin, phymin, phxmax, phymax = metric_calculate.h2Coor(width, height, hp, int(scale))
                        ghxmin, ghymin, ghxmax, ghymax = metric_calculate.h2Coor(width, height, hi, int(scale))
                        total_IoU+=metric_calculate.getIoU(phxmin, phymin, phxmax, phymax,ghxmin, ghymin, ghxmax, ghymax)
                    print total_IoU/cnt
if __name__ == "__main__":
    import json
    import cv2
    import os
    import numpy as np
    import itertools
    from matplotlib import pyplot as plt
    from path_config import *
    import Image
    import xml.etree.cElementTree as ET
    import metric_calculate
    
    ###CONFIG###
    ############
#     print "90 positive test iou"
    motor_50_metri_folder = "/local/wangxin/results/ferrari_gaze/chain_glsvm_et/test_chain_glsvm_motor_50_1e-4/metric"
    jumping_50_metri_folder = "/local/wangxin/results/full_stefan_gaze/chain_glsvm_et/test_chain_glsvm_jumping_90_1e-4/metric"
    jumping_50_metri_folder = "/local/wangxin/results/full_stefan_gaze/chain_glsvm_et/test_chain_glsvm_nonlinear_jumping_50_1e-4/metric"
    
#     metric_file_analyse(metric_folder = "/local/wangxin/results/full_stefan_gaze/glsvm_pos_neg/lsvm_cccpgaze_posneg_inverse_jumping/metric/")
#     metric_file_analyse(metric_folder = jumping_50_metri_folder)
#     iou_hi_hp(metric_folder = jumping_50_metri_folder)
#     gr(metric_folder = jumping_50_metri_folder)
#     visualize_fixations(VOC2012_OBJECT_EYE_PATH)
#     IoU, ratio = correlation_IoU_gaze_ratio(VOC2012_OBJECT_EYE_PATH)
#     IoU=[1,2]
#     ratio=[3,4]
#     plt.scatter(IoU,ratio,c='r', s=1)
#     plt.show() 
    lsvm_region = get_region("/local/wangxin/results/ferrari_gaze/std_et/lsvm_standard/metric_final/30/metric_valtest_30_0.0_1.0E-4_dog.txt")
    glsvm_region = get_region("/local/wangxin/results/ferrari_gaze/std_et/lsvm_cccpgaze_positive_cv/metric_final/30/metric_valtest_0.2_30_0.0_1.0E-4_dog.txt")
    vis_region_proposals(VOC2012_OBJECT_EYE_PATH, lsvm_region, glsvm_region)