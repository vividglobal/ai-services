# Ai Services

## Models
| Model | Description | Link |
| ------ | ------ | ------ | 
| Object detection | Object detection where machine systems and software need to locate objects in an image/scene and identify each object including milk cans, babies, feeding bottles, pacifiers| [![](https://github.githubassets.com/favicons/favicon.png)](https://github.com/vividglobal/yolov5) |
| Text Spotting | Model recognizes text in images is of great importance for many image comprehension tasks such as recognizing milk label names, milk types. It includes two tasks of text detection and recognition.| [![](https://github.githubassets.com/favicons/favicon.png)](https://github.com/vividglobal/pan_pp.pytorch) |
| Multi Task Classification | Use images and text on products to identify products and violations of each brand or product| [![](https://github.githubassets.com/favicons/favicon.png)](https://github.com/vividglobal/multi-task-classification) |
| NLP | Model helps machines understand and effectively perform English-related tasks such as detecting violations on captions.| [![](https://github.githubassets.com/favicons/favicon.png)](https://github.com/vividglobal/nlp) |
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
