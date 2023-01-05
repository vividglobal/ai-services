import base64
import argparse
import sys
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn

sys.path.append('Objectdetection') 

from Objectdetection.yolo.c_utils.utils import object_detect,init_model,draw_box,rtd_detect
from Objectdetection.yolo.c_utils import config as cfg

from project_utils.datasets import LoadStreams, LoadImages
from project_utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from project_utils.plots import colors, plot_one_box
from project_utils.torch_utils import select_device, load_classifier, time_sync
from project_utils.ensemble_labels import display_image, meger_label_branch, word2line, Ensemble, Analyzer
import numpy as np

from project_utils.general import xywh2xyxy, clip_coords
from textSpotting import textSpottingInfer
from classifyText import textClassifyInfer2
from classifyImage import objectClasssifyInfer

model_path_recognition = 'textSpotting/textRecognition/best_accuracy.pth'
craft_path = 'textSpotting/CRAFTpytorch/craft_mlt_25k.pth'

recog_model_dir = 'classifyText/mmocr1/configs/'
pannet_model_path = 'classifyText/newpan/checkpoint/pan.pth.tar'
brand_text_model_path = 'classifyText/textClassify/checkpoints/product/product_classifier_level1.pkl'
step_model_dir_path = "classifyText/textClassify/checkpoints/product/brands/"

brand_image_classifier_model_path = "classifyImage/weights/model_final.h5"
step_image_classifier_model_path = "classifyImage/weights/model_step.pt"
labels_txt_fn = 'classifyImage/labels.txt'
labels_branchs_dict_fn = 'classifyImage/labels_branchs_dict.json'

yolo_model_paths = 'models/weights/best.pt'
rtd_weight = 'models/weights/rtd_best.pt'

keywords_fn = 'keywords.txt'
correct_corpus_fn = 'corpus.txt'


def load_keywords(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
    keywords = []
    for line in lines:
        line = line.strip().split()
        keywords += line
    return keywords

def load_model_yolo(weights,rtd_weight,device):
    return init_model(weights,rtd_weight)

def load_model(device):
    craft_detect, model_recognition = textSpottingInfer.load_model_1(
        model_path_recognition=model_path_recognition,
        craft_path=craft_path,
    )
    mmocr_recog, pan_detect, classifyModel_level1, dict_model = textClassifyInfer2.load_model(
        recog_model_dir=recog_model_dir,
        pannet_model_path=pannet_model_path,
        brand_text_model_path=brand_text_model_path,
        step_model_dir_path=step_model_dir_path,
    )
    (
        chinh_model,
        model_step,
        labels_end,
        labels_branch,
        dict_middle,
        dict_step
    ) = objectClasssifyInfer.load_model(
        weight_path_1=brand_image_classifier_model_path,
        weight_path_2=step_image_classifier_model_path,
        labels_txt_fn=labels_txt_fn,
        labels_branchs_dict_fn=labels_branchs_dict_fn,
    )

    model_object_detect, rtd_model, bb_model = load_model_yolo(
        weights=yolo_model_paths,
        rtd_weight = rtd_weight,
        device=device,
    )

    return (
        (craft_detect, model_recognition),
        (mmocr_recog, pan_detect, classifyModel_level1, dict_model),
        (
            chinh_model,
            model_step,
            labels_end,
            labels_branch,
            dict_middle,
            dict_step,
        ),
        model_object_detect,
        bb_model,
        rtd_model
    )


from PIL import Image
import io

class Infer:
    def __init__(
            self,
            craft_detect, model_recognition,

            mmocr_recog, pan_detect, classifyModel_level1, dict_model,

            chinh_model,
            model_step,
            labels_end,
            labels_branch,
            dict_middle,
            dict_step,

            model_object_detect,
            bb_model,
            rtd_model,
            keywords, spell,
            device,
    ):
        for k, v in locals().items():
            setattr(self, k, v)
            
    def run(self, images):
        craft_detect, model_recognition = self.craft_detect, self.model_recognition

        (
            mmocr_recog, pan_detect, classifyModel_level1, dict_model
        ) = self.mmocr_recog, self.pan_detect, self.classifyModel_level1, self.dict_model,

        (
            chinh_model,
            model_step,
            labels_end,
            labels_branch,
            dict_middle,
            dict_step,
        ) = (
            self.chinh_model,
            self.model_step,
            self.labels_end,
            self.labels_branch,
            self.dict_middle,
            self.dict_step,
        )

        model_object_detect = (self.model_object_detect)
        rtd_model = (self.rtd_model)
        bb_model = (self.bb_model)
        device = self.device
        keywords = self.keywords
        spell = self.spell

        results_end = []
        count=0
        for img in images:

            item = {}

            t1 = time_sync()
            list_box, list_image, list_label,image_ignord = object_detect(img.copy(), model_object_detect, obj_conf=0.4)
            #add list bb
            list_box_bb, list_image_bb, list_label_bb,_ = object_detect(input=img.copy(), yolo_obj_model=bb_model, model_type='bb_detect')
            list_box = list_box + list_box_bb
            list_image = list_image + list_image_bb
            list_label = list_label + list_label_bb
            item['binh_bu'] = False
            item['sua'] = []
            item['rtd_violation'] = []
            item['rtd_list_label'] = []
            text_product = ''
            for indx,img_crop  in enumerate(list_image):
                if(list_label[indx] in ["sua_binh","sua_hop","sua_lon","sua_tui"]):
                    rtd_list_label, rtd_list_image, rtd_code = rtd_detect(img_crop, object_label=list_label[indx], yolo_rtd_model= rtd_model, rtd_conf=0.35)
                    try:
                        result_text_spotting = textClassifyInfer2.spotting_text(pan_detect, craft_detect, mmocr_recog, img_crop)
                        product = textClassifyInfer2.predictKeyword(result_text_spotting.copy(),rtd_list_label)
                        if(list_label[indx] in ["sua_binh","sua_hop","sua_tui"] and product == 'Bellamy_4' ):
                            product='Bellamy_4_cf'
                        for i in result_text_spotting:
                            text = i['text'].lower().replace(' ', '_')
                            text_product+=' '+text
                    except Exception as e:
                        print(e)
                        product = False
                    print(product)
                    print(rtd_code)
                    rtd_code = rtd_code.split('; ')
                    for code in rtd_code:
                        if(code!="" and code not in item['rtd_violation']):
                            item['rtd_violation'].append(code)
                    item['rtd_list_label'] = list(set(item['rtd_list_label']+rtd_list_label))
                    list_label[indx]=product
                    if(product):
                        item['sua'].append(product)
                if(list_label[indx] in ["ti_gia","binh_bu"]):
                    item['binh_bu'] = True
            # Text banner
            result_text = textSpottingInfer.predict(
                image_ignord, craft_detect, model_recognition)
            list_text = []
            final_res = word2line(result_text, image_ignord)
            for res in final_res:
                x1, y1, w, h = res['box']
                x2 = x1+w
                y2 = y1+h
                text = res['text']
                list_text.append((text))
            item['text_banner']= list_text
            item['text_product']= text_product
            # obj_image = draw_box(list_box, list_image, list_label, input_image=img)
            # save_file = f'static/img/{str(int(time.time()))}.jpg'
            # cv2.imwrite(save_file,obj_image)
            # item['image'] = save_file
            results_end.append(item)
            
        return results_end
