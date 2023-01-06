# Ai Services
The content on social networking sites and websites including captions and images. To detect ad violations we combine a variety of techniques: 
| Technique | Model | Description | Link |
| ------ | ------ | ------ | ------ | 
| Multi Task Classification | MobilenetV2, ResNet, biLSTM, ... |  Multi-Task Classification including Text Classification and Image Classification can detect the product in images| [![](https://github.githubassets.com/favicons/favicon.png)](https://github.com/vividglobal/image-classification) [![](https://github.githubassets.com/favicons/favicon.png)](https://github.com/vividglobal/text-classification) [![](https://github.githubassets.com/favicons/favicon.png)](https://github.com/vividglobal/multi-task-classification) |
| Object detection | Yolo5 | Object detection where machine systems and software need to locate objects in an image/scene and identify each object including milk cans, babies, feeding bottles, pacifiers| [![](https://github.githubassets.com/favicons/favicon.png)](https://github.com/ultralytics/yolov5) |
| Text Spotting | PAN ++ | Model recognizes text in images is of great importance for many image comprehension tasks such as recognizing milk label names, milk types. It includes two tasks of text detection and recognition.| [![](https://github.githubassets.com/favicons/favicon.png)](https://github.com/whai362/pan_pp.pytorch) |
| NLP | spaCy  | Model helps machines understand and effectively perform English-related tasks such as detecting violations on captions.| [![](https://github.githubassets.com/favicons/favicon.png)](https://github.com/vividglobal/nlp) |

## Workflow
[![](https://github.com/vividglobal/ai-services/blob/master/diagram/Vivid-Workflow.drawio.png?raw=true)](https://drive.google.com/file/d/1jdjN4Z7Uj368JI-WCavnwV-SG48guAV5/view?usp=sharing)
1. The input is posted on social networking sites and websites, including captions and images.
2. The Text Spotting model extracts text from the images.
3. The NLP model detects the intent of the captions and text in the images.
4. The Object Detection model detects milk cans, babies, feeding bottles, and pacifiers in the images.
5. The Multi-Task Classification model identifies the product in the images.
6. The Conditional Statement determines if there is a violation.
## Clone and setup environment
### Requirments:
```sh
sudo apt-get update
sudo apt-get install g++
sudo apt-get install cmake
```
### setup:
```sh
conda create -n name_envs python==3.7
conda activate name_envs
bash ./connfig.sh
```

### run service
```sh
python run_service.py
```
