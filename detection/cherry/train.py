import torch

from dataset import Dataset, collate_fn
from model import CoffeAI

from vision_utils import utils
from vision_utils.engine import train_one_epoch, evaluate

if __name__ == '__main__':
    # params
    input_dir = '../../data/datasets/initial-dataset-labeled'
    checkpoint = 'checkpoints/pretrained.pt'
    checkout = 'checkpoints/version0.pt'
    epochs = 10

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    dataset = Dataset(input_dir, True)
    dataset_test = Dataset(input_dir, False)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=collate_fn)

    # get the model using our helper function
    coffeai = CoffeAI.from_checkpoint(checkpoint)

    # move model to the right device
    coffeai.to(device)

    # construct an optimizer
    params = [p for p in coffeai.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(coffeai, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(coffeai, data_loader_test, device=device)

    coffeai.to_checkpoint(checkout)
