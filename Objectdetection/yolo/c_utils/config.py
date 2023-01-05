import torch
import torchvision.transforms as transforms

device = 'cpu'

# Transform
img_transforms = transforms.Compose([transforms.Resize([224,224]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(20),
                                     transforms.ToTensor(),
                                     ])
img_transforms_valid = transforms.Compose([transforms.Resize([224,224]),
                                          #  transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           ])
img_transforms_test = transforms.Compose([transforms.Resize([255,255]),
                                          #  transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          ])