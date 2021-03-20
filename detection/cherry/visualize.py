""" Draw Bounding Boxes on (Possibly) Unlabeled Images.

This module is meant for debugging. It loads a pretrained model, predicts 
bounding boxes on random images from the dataset, and creates plots for 
visual inspection.

"""

import os

from PIL import Image

import torch
import torchvision
import numpy as np

from model import CoffeAI

import matplotlib.pyplot as plt
import matplotlib.patches as patches


LABEL_TO_INT = {
    0: 'background',
    1: 'cherry (ripe)',
    2: 'cherry (green)',
}


LABEL_TO_COLOR = {
    0: 'k',
    1: 'b',
    2: 'r'
}


def label_to_text(label: int):
    """

    """
    text = LABEL_TO_INT.get(label, None)
    if text is None:
        return 'undefined'
    else:
        return text


def label_to_color(label: int):
    """

    """
    color = LABEL_TO_COLOR.get(label, None)
    if color is None:
        return 'k'
    else:
        return color


def plot_image_with_bounding_boxes(img: Image, predictions: dict):
    """

    """
    #
    boxes = predictions['boxes']
    labels = predictions['labels']

    #
    fig, ax = plt.subplots()
    ax.imshow(img)
    #
    for label, box in zip(labels, boxes):
        label = int(label)
        text_label = label_to_text(label)
        x = box[0]
        y = box[1]
        w = box[2] - box[0]
        h = box[3] - box[1]
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=1,
            edgecolor=label_to_color(label),
            facecolor='none',
        )
        ax.add_patch(rect)
        ax.text(x, y, text_label, fontsize=6, color=label_to_color(label))
    
    ax.axis('off')
    return fig


def filter_low_confidence(predictions: dict, min_confidence: float):
    """

    """
    scores = predictions['scores']
    predictions['boxes'] = predictions['boxes'][scores > min_confidence]
    predictions['labels'] = predictions['labels'][scores > min_confidence]
    predictions['scores'] = predictions['scores'][scores > min_confidence]
    return predictions


def filter_nms(predictions: dict, iou_threshold: float):
    """

    """
    nms = torchvision.ops.nms(predictions['boxes'], predictions['scores'], iou_threshold)
    predictions['boxes'] = predictions['boxes'][nms]
    predictions['labels'] = predictions['labels'][nms]
    predictions['scores'] = predictions['scores'][nms]
    return predictions



if __name__ == '__main__':

    # params
    input_dir = '../../data/datasets/raw-dataset'
    output_dir = 'outputs'
    checkpoint = 'checkpoints/version0.pt'

    min_confidence = 0.
    iou_threshold = 0.


    # helpers
    to_tensor = torchvision.transforms.ToTensor()

    # get random images
    filenames = os.listdir(input_dir)
    filenames = [os.path.join(input_dir, fname) for fname in filenames]

    np.random.seed(1234)
    filenames = np.random.permutation(filenames)[:20]

    # load model
    coffeai = CoffeAI.from_checkpoint(checkpoint)
    coffeai.eval()

    # generate images
    with torch.no_grad():
        for i, fname in enumerate(filenames):
            # load image
            img = Image.open(fname)
            # show to coffeai
            tensor_img = to_tensor(img)
            predictions = coffeai([tensor_img])[0]
            predictions = filter_low_confidence(predictions, min_confidence)
            predictions = filter_nms(predictions, iou_threshold)
            # plot predictions on image
            fig = plot_image_with_bounding_boxes(img, predictions)
            fig.savefig(os.path.join(output_dir, f'bbox_example_{i}.png'))


        
