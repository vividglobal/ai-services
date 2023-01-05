from c_utils.utils import *
from c_utils import config as cfg
# Step:
# 1.Khai báo weight path
# 2.input image folder path & save folder path
# 3.init model - input weight của object detect và rtd để tạo 2 model object detect (yolo_model) và rtd
# 4.output của object detect là ảnh gốc có bbox, list ảnh của object có trong ảnh gốc và list label có stt tương ứng
# 5.trong list ảnh object nếu có object nào là sữa bình or sữa hộp thì sửa dụng model rtd
# 6.output của rtd detect là ảnh gốc có bbox, list ảnh của object có trong ảnh gốc và list label có stt tương ứng
# 7.function draw_box dùng để vẽ box lên ảnh gốc (set save_result để lưu, visualize để xuất ảnh - default cả 2 = False)
yolo_weight = r'E:\PycharmProjects\Milk_Infringing_Advertising\weight\object_weight.pt'
rtd_weight = r'E:\PycharmProjects\Milk_Infringing_Advertising\weight\rtd_best.pt'

save_path = r'E:\data_test\Object Detect classification 23.06\Object Detect classification 23.06\results_Object_Detect_23.06\obj'
folder = r'E:\data_test\Object Detect classification 23.06\Object Detect classification 23.06\RTD'

if __name__ == '__main__':
    yolo_model, rtd_model = init_model(yolo_weight, rtd_weight)
    for image in os.listdir(folder):
        image_path = os.path.join(folder, image)
        # create list (default obj_conf=0.33 and customizable)
        image = cv2.imread(image_path)
        list_box, list_image, list_label,image_ignord = object_detect(input=image, yolo_obj_model=yolo_model)

        # draw box and save image (optional)
        # draw_box(list_box, list_image, list_label, save_path=save_path, image_path=image_path,
        #          save_result=True, visualize=True)
        end_image = draw_box(list_box, list_image, list_label, image_path=image_path)
        cv2.imshow("Image", end_image)
        cv2.waitKey(0)
        count = 0
        for num, label in enumerate(list_label):
            if label == "sua_binh" or label == "sua_hop":
                # create list (default rtd_conf=0.6 and customizable)
                rtd_list_box, rtd_list_image, rtd_list_label = rtd_detect(list_image[num], yolo_rtd_model=rtd_model)

                # draw box rtd and save image (optional) - set is_rtd=True & input object image
                rtd_image = draw_box(rtd_list_box, rtd_list_image, rtd_list_label, image_path=image_path,
                         input_image=list_image[num], is_rtd=True)
                count += 1


