import scipy.io as sio
import math
import os.path
VOC2012_OBJECT_CATEGORIES = ["bicycle", "diningtable", "cow", "horse" ,"sofa","boat","cat","aeroplane","dog", "motorbike",]
for c in VOC2012_OBJECT_CATEGORIES:
    mat_contents = sio.loadmat('/local/wangxin/Data/ferrari_data/etData/etData_%s.mat'%c)
    root="/local/wangxin/Data/ferrari_gaze/triplet_gaze/%s/"%c
    if not os.path.exists(root):
        os.makedirs(root)
    img_names = mat_contents['etData']['filename']
    fixations=mat_contents['etData']['fixations']
    img_num,_= fixations.shape
    for i in range(img_num):
        val = fixations[i,0]
        subj = val["imgCoord"]
        f = open(root+img_names[i,0][0]+".trigaze","w")
        _, subj_num = subj.shape
        for j in range(subj_num):
            fixs = subj[0,j]
            fixR = fixs['fixR']
            fixR_time =fixR[0,0]['time']
            fixR_pos =fixR[0,0]['pos']
            fixL = fixs['fixL']
            fixL_time =fixL[0,0]['time']
            fixL_pos =fixL[0,0]['pos']
            
            tsr =  fixR_time[0][0][:,1] - fixR_time[0][0][:,0]
            tsl =  fixL_time[0][0][:,1] - fixL_time[0][0][:,0]
    #         print fixR_pos
            
#             fs =  (fixR_pos + fixL_pos)/2
        
    #         print fixL_time
    #         print fixL_pos
            print len(tsr)
            for m in range(len(tsr)):
                if  not (math.isnan(float(fixR_pos[0][0][m,:][0])) and
                    not math.isnan(float(fixR_pos[0][0][m,:][1]))):
                    f.write("%f,%f,%f\n"%(float(fixR_pos[0][0][m,:][0]), float(fixR_pos[0][0][m,:][1]), float(tsr[m])))
                elif not (math.isnan(float(fixL_pos[0][0][m,:][0])) and
                    not math.isnan(float(fixL_pos[0][0][m,:][1]))):
                    f.write("%f,%f,%f\n"%(float(fixL_pos[0][0][m,:][0]), float(fixL_pos[0][0][m,:][1]), float(tsl[m])))

                else:
                    continue
        f.close()
                    
