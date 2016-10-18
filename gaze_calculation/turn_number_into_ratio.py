VOC2012_OBJECT_CATEGORIES = ["bicycle", "diningtable", "cow", "horse" ,"sofa","boat","cat","aeroplane","dog", "motorbike",]

root = "/local/wangxin/Data/ferrari_gaze/gaze_number_weighted"
new_root="/local/wangxin/Data/ferrari_gaze/gaze_ratio_weighted"
import os
import math
for c in VOC2012_OBJECT_CATEGORIES:
    for full_f in os.listdir(root + '/' + c + '/1/'):
        
        for scale in ['1','4','9','16','25','36','49','64']:
#         for scale in ['36']:
            maxnum=-1
            for i in range(int(math.sqrt(int(scale)))):
                for j in range(int(math.sqrt(int(scale)))):
                    fileroot='_'.join(full_f.split('_')[:-2])
                    f = open(root+'/'+c+'/'+scale+'/'+fileroot+'_'+str(i)+'_'+str(j)+'.txt')
                    given_number=f.readline()
                    if float(given_number)>maxnum:
                        maxnum = float(given_number)
            for i in range(int(math.sqrt(int(scale)))):
                for j in range(int(math.sqrt(int(scale)))):
                    fileroot='_'.join(full_f.split('_')[:-2])
                    f = open(root+'/'+c+'/'+scale+'/'+fileroot+'_'+str(i)+'_'+str(j)+'.txt')
                    given_number=f.readline()
                    new_folder = new_root +'/'+ c + '/'+scale
                    if not os.path.exists(new_folder):
                        os.makedirs(new_folder)
                    new_f = open(new_folder+'/'+fileroot+'_'+str(i)+'_'+str(j)+'.txt','w')                    
                    new_f.write(str(float(given_number)/float(maxnum)))
                    new_f.close()
                    f.close()
                    
                    