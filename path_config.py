"""PASCAL VOC 2012 DATASET"""
VOC2012_TRAIN_ROOT = "/local/wangxin/Data/VOCdevkit_trainset/VOC2012/"
VOC2012_TRAIN_IMAGES  = VOC2012_TRAIN_ROOT + "JPEGImages/"
VOC2012_TRAIN_ANNOTATIONS = VOC2012_TRAIN_ROOT + "Annotations/"
VOC2012_TRAIN_LIST = VOC2012_TRAIN_ROOT + "ImageSets/"

"""STEFAN PASCAL VOC2012 ACTION EYE-TRACKING DATASET"""

VOC2012_ACTION_ROOT = "/local/wangxin/Data/gaze_voc_actions_stefan/"
VOC2012_ACTION_ORIGIN_SPLIT = VOC2012_TRAIN_ROOT+"ImageSets/Action/"
VOC2012_ACTION_EYE_PATH = VOC2012_ACTION_ROOT+"samples/"
VOC2012_ACTION_EYE_ACTION_JSON_PATH = VOC2012_ACTION_ROOT+"full_train_gazes/"
VOC2012_ACTION_EYE_CONTEXT_JSON_PATH = VOC2012_ACTION_ROOT+"train_gazes_context/"
VOC2012_ACTION_TRAIN_LIST = VOC2012_ACTION_ROOT+"action_train_image_list"
VOC2012_ACTION_ETLOSS_CONTEXT = VOC2012_ACTION_ROOT + "ETLoss_ratio_context/"
VOC2012_ACTION_ETLOSS = VOC2012_ACTION_ROOT + "ETLoss_ratio/"
VOC2012_ACTION_METRIC_ROOT = "/home/wangxin/ovelapping_files/stefan/"

VOC2012_ACTION_CATEGORIES= ['jumping', 'phoning', 'playinginstrument', 'reading', 'ridingbike', 'ridinghorse', 'running', 'takingphoto', 'usingcomputer', 'walking']

VOC2012_ACTION_ACTION_SUBJS  = ["006","007","008","009","010","011","018","020"] 
VOC2012_ACTION_VALIDE_SUBJS = ["006","007","008","009","010","011","018"]
VOC2012_ACTION_VALIDE_SUBJS_CONTEXT = ["015","017","021","022"]


"""PAPA. PASCAL VOC2012 OBJECT EYE-TRACKING DATASET"""
VOC2012_OBJECT_ROOT = "/local/wangxin/Data/ferrari_gaze/"
VOC2012_OBJECT_CATEGORIES = ["bicycle", "diningtable", "cow", "horse" ,"sofa","boat","cat","aeroplane","dog", "motorbike",]
VOC2012_OBJECT_EYE_PATH = "/local/wangxin/Data/ferrari_gaze/triplet_gaze/"
VOC2012_OBJECT_ETLOSS = VOC2012_OBJECT_ROOT + "ETLoss_ratio_weighted/"
VOC2012_OBJECT_METRIC_ROOT = "/local/wangxin/results/ferrari_gaze/std_et/"
