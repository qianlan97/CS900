# SPDX-License-Identifier: LGPL-3.0-or-later
import torch
from torchvision import models
from PIL import Image
import tensorflow as tf

output_filename = "alexnet-pretrained.pt"

try:
    alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
except AttributeError:
    # older versions of torchvision (less than 0.13.0) use below now-deprecated parameter
    alexnet = models.alexnet(pretrained=True)

torch.save(alexnet, output_filename)

print("Pre-trained model was saved in \"%s\"" % output_filename)

output_filename = "resnet18-pretrained.pt"

try:
    resnet = models.resnet18(pretrained=True)
except AttributeError:
    # older versions of torchvision (less than 0.13.0) use below now-deprecated parameter
    resnet = models.resnet18(pretrained=True)

torch.save(resnet, output_filename)

print("Pre-trained model was saved in \"%s\"" % output_filename)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
image_index = 0
image = x_test[image_index]
image_path = 'cifar10_test_image.png'
image = Image.fromarray(image)
image.save(image_path)