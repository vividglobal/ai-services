import os
import torch
import cv2
import numpy as np


def init_model(object_detect_weight, rtd_weight):
    yolo_obj_model = torch.hub.load('ultralytics/yolov5', 'custom', path=object_detect_weight, force_reload=True)
    yolo_rtd_model = torch.hub.load('ultralytics/yolov5', 'custom', path=rtd_weight, force_reload=True)
    #add yolo_bb_model
    yolo_bb_model = torch.hub.load('ultralytics/yolov5', 'custom', path=object_detect_weight, force_reload=True)
    return yolo_obj_model, yolo_rtd_model, yolo_bb_model


def yolo_run(image, model, model_type="object_detect", rtd_conf=0.2, obj_conf=0.3, bb_conf=0.38):
    # add bb_conf = 0.38
    list_box = []
    list_image = []
    list_label = []
    image_ignord = image.copy()
    output = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_type == 'rtd':
        model.conf = rtd_conf
        model.classes = [0, 1, 8, 9, 11, 12, 13, 14, 15, 2, 10]
    elif model_type == 'brand':
        model.conf = rtd_conf
        model.classes = [2, 3, 4, 5, 6, 7, 10]
    #Chage model.classes từ [0, 1, 2, 3, 5, 6] thành [0, 2, 3, 5, 6]
    elif model_type == 'object_detect':
        model.conf = obj_conf
        model.classes = [0, 2, 3, 5, 6]
    #add conditional model_type == 'bb_detect'
    elif model_type == 'bb_detect':
        model.conf = bb_conf
        model.classes = [1]

    model.to(device)
    #add model.agnostic, model.ampz
    model.agnostic = True
    model.amp = True
    #add conditional model_type == 'bb_detect' => results = model(..., augment=True)
    if model_type == 'bb_detect':
        results = model(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), size=640, augment=True)  # includes NMS
    else:
        results = model(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), size=640)  # includes NMS
    label_files = results.pred[0].tolist()
    names = results.names
    for num, file in enumerate(label_files):
        boxes = file[:4]
        label = file[-1]
        ymin = round((float(boxes[0])))
        xmin = round((float(boxes[1])))
        ymax = round((float(boxes[2])))
        xmax = round((float(boxes[3])))
        iloc = [xmin, ymin, xmax, ymax]

        cropped_image = image[xmin:xmax, ymin:ymax].copy()
        ignord = np.zeros_like(cropped_image)
        if model_type != 'bb_detect':
            image_ignord[xmin:xmax, ymin:ymax]=ignord
        list_box.append(iloc)
        list_image.append(cropped_image)
        list_label.append(names[int(label)])
        output = [list_box, list_image, list_label]
    return output,image_ignord


def draw_box(list_box, list_image, list_label, image_path=None, input_image=None):
    if input_image is not None:
        image = input_image
    else:
        image = cv2.imread(image_path)
    for num, img in enumerate(list_image):
        xmin, ymin, xmax, ymax = list_box[num]
        image = cv2.rectangle(image, (ymin, xmin), (ymax, xmax), (36, 255, 12), 1)
        label = list_label[num]
        cv2.putText(image, str(label), (ymin, xmin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return image


# def draw_box(list_box, list_image, list_label, save_path=None, image_path=None, input_image=None,
#              count=None, save_result=False, visualize=False, is_rtd=False):
#     if is_rtd:
#         image = input_image
#     else:
#         image = cv2.imread(image_path)
#     for num, img in enumerate(list_image):
#         xmin, ymin, xmax, ymax = list_box[num]
#         image = cv2.rectangle(image, (ymin, xmin), (ymax, xmax), (36, 255, 12), 1)
#         label = list_label[num]
#         cv2.putText(image, str(label), (ymin, xmin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
#     if is_rtd:
#         folder = os.path.join(save_path, 'rtd')
#         if not os.path.isdir(folder):
#             os.makedirs(folder)
#         name = os.path.join(folder, str(count) + "_" + os.path.basename(image_path))
#         if save_result:
#             cv2.imwrite(name, image)
#         if visualize:
#             try:
#                 cv2.imshow("Image", image)
#                 cv2.waitKey(0)
#             except:
#                 from google.colab.patches import cv2_imshow
#                 cv2_imshow(image)
#     if is_rtd:
#         folder = os.path.join(save_path, 'rtd')
#         if not os.path.isdir(folder):
#             os.makedirs(folder)
#         name = os.path.join(folder, str(count) + "_" + os.path.basename(image_path))
#         if save_result:
#             cv2.imwrite(name, image)
#         if visualize:
#             try:
#                 cv2.imshow("Image", image)
#                 cv2.waitKey(0)
#             except:
#                 from google.colab.patches import cv2_imshow
#                 cv2_imshow(image)
#     else:
#         if save_result:
#             save_obj(save_path, image_path, image)
#         if visualize:
#             try:
#                 cv2.imshow("Image", image)
#                 cv2.waitKey(0)
#             except:
#                 from google.colab.patches import cv2_imshow
#                 cv2_imshow(image)
#     return image, list_image, list_label


def save_obj(save_path, image_path, image):
    if save_path is None:
        name = image_path + "_pred.jpg"
        cv2.imwrite(name, image)
        print(name)
    else:
        name = os.path.join(save_path, os.path.basename(image_path))
        cv2.imwrite(name, image)
        print(name)
    print("-------------------")


def object_detect(input, yolo_obj_model, obj_conf=0.33, model_type='object_detect'):
    #add model_type='object_detect'
    if type(input) == str:
        image = cv2.imread(input)
    image = input
    list_box = []
    list_image = []
    list_label = []
    #add model_type='object_detect'
    yolo_output,image_ignord = yolo_run(image, yolo_obj_model, obj_conf=obj_conf, model_type=model_type)
    if len(yolo_output) != 0:
        list_box, list_image, list_label = yolo_output
    return list_box, list_image, list_label,image_ignord



def rtd_detect(image, object_label, yolo_rtd_model, rtd_conf=0.6):
    list_box = []
    list_label = []
    list_image = []
    yolo_output = []
    code = ''
    if object_label == "sua_binh" or object_label == "sua_hop":
        yolo_output,image_ignord = yolo_run(image, yolo_rtd_model, model_type='rtd', rtd_conf=rtd_conf)
    elif object_label == "sua_lon" or object_label == "sua_tui":
        yolo_output,image_ignord = yolo_run(image, yolo_rtd_model, model_type='brand', rtd_conf=rtd_conf)
    if len(yolo_output) != 0:
        list_box, list_image, list_label = yolo_output
        print(list_label)
        for i in range(len(list_label)):
            if list_label[i] == "Peadiasure":
                code = "5.1"
                list_label[i] = "Pediasure"
            if list_label[i] == "Momcare":
                code = "CMF-PW"
            if list_label[i] == "Aptamil":
                code = "WHA58.32; 9.1 + 9.2,4.2"
            if list_label[i] == "Enfamil":
                code = "WHA58.32; 9.1 + 9.2,4.2"
            if list_label[i] == "Similac360":
                code = "WHA58.32; 9.1 + 9.2,4.2"
                list_label[i] = "Similac"
            if list_label[i] == "SimilacPro":
                code = "WHA58.32; 9.1 + 9.2,4.2"
                list_label[i] = "Similac"
            if list_label[i] == "Similac":
                code = "WHA58.32; 9.1 + 9.2,4.2"
            if list_label[i] == "SimilacOrganic":
                code = "WHA58.32 + WHA61.20-Label_contam; WHA58.32; 9.1 + 9.2,4.2"
                list_label[i] = "Similac"
            if list_label[i] == "Gerber good start":
                code = "WHA58.32"
                list_label[i] = "GEBER"
            if list_label[i] == "Karihome_1":
                code = "5.1"
                list_label[i] = "Karihome"
            if list_label[i] == "Karihome_2":
                code = "5.1"
                list_label[i] = "Karihome"
            if list_label[i] == "Karihome_3":
                code = "9.1 + 9.2,4.2; WHA58.32"
                list_label[i] = "Karihome"
            if list_label[i] == "baby_gerber":
                # if "Gerber" in list_label:
                list_label[i] = "Gerber"
                code = f"WHO Rec. 4"

    return list_label, list_image, code