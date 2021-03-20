""" Cherry Detector Model CoffeAI

This module contains the detector model CoffeAI. It's a FasterRCNN pretrained
on COCO with an altered prediction head. The class provides additional, 
simplified interfaces for saving and loading checkpoints.

"""

import torch
import torch.nn as nn

import torchvision
from torchvision.models import detection


class CoffeAI(nn.Module):
    """

    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):

        super(CoffeAI, self).__init__()

        self.num_classes = num_classes + 1 # add background class
        self.model = detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = \
            detection.faster_rcnn.FastRCNNPredictor(in_features, self.num_classes)

    @classmethod
    def from_checkpoint(cls, checkpoint: str):
        """

        """
        model = cls(pretrained=False)
        model.model = torch.load(checkpoint)
        return model
        

    def to_checkpoint(self, checkpoint: str):
        """

        """
        torch.save(self.model, checkpoint)


    def forward(self, *args):
        """

        """
        return self.model(*args)



if __name__ == '__main__':
    # create model and save it to checkpoints
    model = CoffeAI(num_classes=2)
    model.to_checkpoint('./checkpoints/pretrained.pt')
