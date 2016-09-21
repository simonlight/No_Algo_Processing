import os
import itertools
import path_config
import re
#generate example_files for training and validation set
def contains_object(cls, item):
    new_item = item.replace('_', '-')
    if cls in new_item:
        return "1"
    else:
        return "0"
def item_class(item):
    return '-'.join(item.split('_')[:-1])
#param: val|train|test
#refpath: [train_file, val_file, trainval_file ]path
#cls: class name 
#file_typ: "train", "val", "trainval"
#scale: 30-100
#vm: validated images. Some images in ferrari's data are not in the original exp_type
#exp_type: experiment type: 'fuul' 'reduit' 'ground'
def generate(all_name, cls, scale):
    example_list_dir = "/local/wangxin/Data/UPMC_Food_Gaze_20/example_files/"+str(scale)+"/"
            
    if not os.path.exists(example_list_dir):
        os.makedirs(example_list_dir)
    
#     train_f = open('_'.join([example_list_dir+cls, 'train', 'scale', str(scale), 'matconvnet_m_2048_layer_20.txt']),'w')
#     val_f = open('_'.join([example_list_dir+cls, 'val', 'scale', str(scale), 'matconvnet_m_2048_layer_20.txt']),'w')
#     trainval_f = open('_'.join([example_list_dir+cls, 'trainval', 'scale', str(scale), 'matconvnet_m_2048_layer_20.txt']),'w')
#     test_f = open('_'.join([example_list_dir+cls, 'test', 'scale', str(scale), 'matconvnet_m_2048_layer_20.txt']),'w')
    full_f = open('_'.join([example_list_dir+cls, 'full', 'scale', str(scale), 'matconvnet_m_2048_layer_20.txt']),'w')
    #image quantity
#     train_f.write("1600\n")
#     val_f.write("200\n")
#     trainval_f.write("1800\n")
#     test_f.write("200\n")
    full_f.write("2000\n")
    suffix = ['_'+str(i)+'_'+str(j) for i,j in itertools.product(range((100-scale)/10+1),range((100-scale)/10+1))]
    for cnt, item in enumerate(all_name):
        item_cls = item_class(item)
        content = item
        content += ' ' + contains_object(cls, item)
        content += ' ' + str(int(11 - 0.1 * scale) ** 2)
        for suf in suffix:
            content += ' ' + '/local/wangxin/Data/UPMC_Food_Gaze_20/vgg-m-2048_features/'+item_cls+'/'+str(scale)+'/'+item +suf+'.txt'
        if cnt<80*20:
#             train_f.write(content+'\n')
#             trainval_f.write(content+'\n')
            full_f.write(content+'\n')
        if 80*20<=cnt<90*20:
#             val_f.write(content+'\n')
#             trainval_f.write(content+'\n')
            full_f.write(content+'\n')
        if 90*20<=cnt<100*20:
#             test_f.write(content+'\n')
            full_f.write(content+'\n')
if __name__ =="__main__":
    #test/train files root
    categories = ["apple-pie",
"bread-pudding",
"beef-carpaccio",
"beet-salad",
"chocolate-cake",
"chocolate-mousse",
"donuts",
"beignets",
"eggs-benedict",
"croque-madame",
"gnocchi",
"shrimp-and-grits",
"grilled-salmon",
"pork-chop",
"lasagna",
"ravioli",
"pancakes",
"french-toast",
"spaghetti-bolognese",
"pad-thai"]
def get_all_name():
    all_name=[]
    temp_all_name=[]
    for cls in categories:
        for elem in os.listdir("/local/wangxin/Data/UPMC_Food_Gaze_20/vgg-m-2048_features/"+cls+"/100/"):
            temp_all_name.append('_'.join(elem.split('_')[:-2]))
    print "total examples:"+str(len(temp_all_name))
    print temp_all_name
    split=20
    for s in range(split):
        all_name.extend(temp_all_name[s*100:s*100+80])
    for s in range(split):
        all_name.extend(temp_all_name[s*100+80:s*100+90])
    for s in range(split):
        all_name.extend(temp_all_name[s*100+90:s*100+100])
    print all_name
    return all_name
all_name=get_all_name()
for scale in range(100, 29, -10):
    print scale
    for cls in categories:
        generate(all_name, cls, scale)
            
