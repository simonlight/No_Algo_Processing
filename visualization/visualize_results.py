def res_file_2_dict(ap_results):
    res = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    for line in ap_results:
        category, tradeoff, scale, lbd, epsilon, ap_train, ap_val, ap_test = [i.split(":")[1] for i in line.strip().split()]
        res[scale][tradeoff][category]= [float(ap_train), float(ap_val), float(ap_test)]
    return res
    

def plot_res(res, res_typ):
#     for scale in ap_res.keys():
    for scale in ['90','80','70','60', '50', '40', '30']:

        result_name = scale
        x_axis = res[scale].keys()
        tradeoff_cv = [0.0,0.1,0.5,1.0,1.5,2.0,5.0,10.0]
        y_train = [0]*len(tradeoff_cv)
        y_val = [0]*len(tradeoff_cv)
        y_test = [0]*len(tradeoff_cv)
        for tradeoff in tradeoff_cv:
            y_ap_all = res[scale][str(tradeoff)]
#             print tradeoff, y_ap_all
            print scale, tradeoff, np.sum(y_ap_all.values(), axis=0) / len(y_ap_all.values())
#             print np.sum(y_ap_all.values(), axis=0)
            ap_train, ap_val, ap_test = np.sum(y_ap_all.values(), axis=0) / len(y_ap_all.values())
            y_train[tradeoff_cv.index(tradeoff)] = ap_train
            y_val[tradeoff_cv.index(tradeoff)] = ap_val
            y_test[tradeoff_cv.index(tradeoff)] = ap_test
        
        
        
        plt.figure(figsize=(8,4))
        plt.plot(tradeoff_cv,y_train,label="train "+res_typ,color="red",linewidth=2)
        plt.plot(tradeoff_cv,y_val,label="validation "+res_typ,color="blue",linewidth=2)
        plt.plot(tradeoff_cv,y_test,label="test "+res_typ,color="green",linewidth=2)
        plt.xlabel("Tradeoff")
        plt.ylabel(res_typ)
        plt.ylim(min(min(y_train), min(y_val),min(y_test)),max(max(y_train), max(y_val),max(y_test)))
        plt.title(res_typ+" of scale:%s, gain of test:%4.2f %%"%(scale,(y_test[y_val.index(max(y_val))]-y_test[0])*100))
        plt.axvline(x=tradeoff_cv[y_val.index(max(y_val))], color= 'black', linestyle='dashed')
        plt.legend(loc='best',fancybox=True,framealpha=0.5)
        plt.show()

if __name__ == "__main__":
    
    import collections
    import matplotlib.pyplot as plt
    import csv
    import numpy as np
    import os
    import visualize_fixations_ferrari as vff
#     import visualize_fixations_stefan as vfs
#     ap_results = open("/local/wangxin/results/ferrari_gaze/std_et/java_std_et_basic_loss/ap_summary.txt")
#     ap_res = res_file_2_dict(ap_results)
#     plot_res(ap_res, "AP")
#     
#     ap_results = open("/local/wangxin/results/full_stefan_gaze/std_et/java_std_et_basic_loss/ap_summary.txt")
#     ap_res = res_file_2_dict(ap_results)
#     plot_res(ap_res, "AP")

    #lsvm ferrai
    ap_results = open("/local/wangxin/results/ferrari_gaze/std_et/lsvm_posneg_loss/ap_summary.txt")
    ap_res = res_file_2_dict(ap_results)
    plot_res(ap_res, "AP")

     
#     detection_folder = "/local/wangxin/results/ferrari_gaze/std_et/java_std_et_basic_loss/metric/"
#     detection_res, gr_res = vff.metric_file_analyse(detection_folder)
#     print detection_res
#     plot_res(detection_res, "detection (ALL images)")
#     plot_res(gr_res, "gaze ratio (ALL images)")

        
            
#     classes=['cat', 'dog', 'bicycle', 'motorbike', 'boat', 'aeroplane', 'horse', 'cow', 'sofa', 'diningtable']
#     my_xticks = classes
#     for scale in res:
#         et=[0]*10
#         gd=[0]*10
#         std=[0]*10
#         for exp_type in res[scale]:
#             if exp_type == "ground":
#                 for cls in res[scale][exp_type]:
#                     gd[classes.index(cls)]=eval(res[scale][exp_type][cls].strip())
#             if exp_type == "reduit_allbb":
#                 for cls in res[scale][exp_type]:
#                     et[classes.index(cls)]=eval(res[scale][exp_type][cls].strip())
#             if exp_type == "reduit_singlebb":
#                 for cls in res[scale][exp_type]:
#                     std[classes.index(cls)]=eval(res[scale][exp_type][cls].strip())
#         print gd
#         print std
#         plt.xticks(x, my_xticks, rotation=-30)
#         plt.xlabel("class name")
#         plt.ylabel("average IoU")
#         plt.plot(x,gd, color='r', label= cast_name("ground"))
#         plt.plot(x,et, color='g',label= cast_name("reduit_allbb"))
#         plt.plot(x,std, color='b', label= cast_name("reduit_singlebb"))
#         title="scale:"+str()
#         plt.legend()
#         plt.grid()
#         plt.title("scale:"+scale+" avg(std)="+str(sum(std)/len(std))[:5]+" avg(et)="+str(sum(et)/len(et))[:5]+ " avg(gd)="+str(sum(gd)/len(gd))[:5])
#         #plt.show()
#         plt.savefig("/home/wang/"+scale+"_iou")
#         plt.clf()
#         
#         with open('/home/wang/'+str(scale)+'.csv', 'wb') as csvfile:
#             spamwriter = csv.writer(csvfile, delimiter=',',
#                                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
#             spamwriter.writerow(['scale='+scale,'mean']+classes)
#             spamwriter.writerow(['std']+[str(sum(std)/len(std))[:5]]+[str(c)[:5] for c in std])
#             spamwriter.writerow(['ground']+[str(sum(gd)/len(gd))[:5]]+[str(c)[:5] for c in gd])
#             spamwriter.writerow(['et']+[str(sum(et)/len(et))[:5]]+[str(c)[:5] for c in et])
#             
