import argparse
import copy
import json
import sys
import time
from collections import OrderedDict
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms



parser = argparse.ArgumentParser(description='PyTorch Image Predicter')
parser.add_argument('image', type=str,
                    help='path to image to process and predict')
parser.add_argument(
    'checkpoint', type=str, help='the file path to your model checkpoint')
parser.add_argument('-k', '--top_k', type=int, default=1,
                    help='the top K most likely classes, default 1')
parser.add_argument('--category_names', default='', type=str,
                    help='category file, json format')
parser.add_argument('-g', '--gpu', default=False, action='store_true',
                    help='predict using GPU')


def main():
    args = parser.parse_args()
    device = torch.device("cuda:0" if (
        torch.cuda.is_available() and args.gpu) else "cpu")

    print("=> processing input")
    normalized_image = process_image(args.image)

    print("=> loading model")
    model, arch, class_idx = load_model(args.checkpoint, device)

    print("=> making top {} predictions using {}".format(args.top_k, device))
    time.sleep(1)
    classes, probs = predict(normalized_image, model,
                             args.top_k, device, args.category_names, class_idx)

    print_predictions(classes, probs)

    pass


def load_model(checkpoint, device):

    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]) and name.startswith('vg'))

    if device == 'cpu':
        model_trained = torch.load(checkpoint, map_location=device)
    else:
        model_trained = torch.load(checkpoint)

    arch = model_trained['arch']
    class_idx = model_trained['class_to_idx']

    if arch in model_names:
        load_model = eval('models.' + arch + '(pretrained=True)')
    else:
        print('{} architecture not recognized. Supported arguments: {}'.format(
            arch, ', '.join(model_names)))
        sys.exit()

    for par in load_model.parameters():
        par.requires_grad = False

    load_model.classifier = model_trained['classifier']
    load_model.load_state_dict(model_trained['state_dict'])

    return load_model, arch, class_idx


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    image = Image.open(image_path, 'r')

    # image resizing

    if image.size[0] > image.size[1]:
        image.thumbnail((100000, 256))  # abritrary maximums
    else:
        image.thumbnail((256, 100000))

    # image cropping, calculating margins needed

    left = (image.width - 224) / 2
    bottom = (image.height - 224) / 2
    right = left + 224
    top = bottom + 224

    image = image.crop((left, bottom, right, top))

    # image normalisation

    image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])  # this was provided
    stdev = np.array([0.229, 0.224, 0.225])  # this was provided
    image = (image - mean) / stdev  # normalised

    # color channels need to be in first dimension, per PyTorch requirements

    image = image.transpose((2, 0, 1))

    return image


def predict(image, model, topk, device, category_names, class_idx):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img_tensor = torch.from_numpy(image).type(
        torch.FloatTensor)  # convert img to numpy

    model_input = img_tensor.unsqueeze(0)  # makes batch size one

    with torch.no_grad():
        output = model.forward(model_input)
        res = torch.exp(output).data.topk(topk)

    classes = np.array(res[1][0], dtype=np.int)
    probabilities = Variable(res[0][0]).data

    if len(category_names) > 0:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)

        mapped = {}
        for k in class_idx:
            mapped[cat_to_name[k]] = class_idx[k]
        # inverts dictionary
        mapped = {v: k for k, v in mapped.items()}
        classes = [mapped[x] for x in classes]
        probs = list(probabilities)
    else:
        class_idx = {v: k for k, v in class_idx.items()}
        classes = [class_idx[x] for x in classes]
        probs = list(probabilities)

    return classes, probs


def print_predictions(classes, probs):
    predictions = list(zip(classes, probs))
    for i in range(len(predictions)):
        print('=> {} : {:.3%}'.format(predictions[i][0], predictions[i][1]))
    pass

if __name__ == '__main__':
    main()
