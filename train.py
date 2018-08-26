import argparse
import copy
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch import optim
from torch.optim import lr_scheduler


def get_user_args():

    model_names = sorted(name for name in models.__dict__
                         if name.islower() and name.startswith('vg') and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(
        description='PyTorch Model Builder - Brenner Swenson')
    parser.add_argument(
        'data', type=str, help='REQUIRED: path to dataset')

    parser.add_argument('-s',
                        '--save_dir', type=str, default='', help='REQUIRED: path to save model checkpoints')

    parser.add_argument('-a', '--arch', default='vgg11',
                        choices=model_names, help='model architecture: ' +
                        ' | '.join(model_names) + ' (default: vgg11)')

    parser.add_argument('-hu', '--hidden_units', default=4096, type=int,
                        help='hidden layers in model architecture (default 4096)')

    parser.add_argument('-e', '--epochs', default=3, type=int,
                        help='number of total epochs to run (default: 3)')

    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='batch size (default: 32)')

    parser.add_argument('-lr', '--learning_rate', default=0.001,
                        type=float, help='initial learning rate (default: 0.001)')

    parser.add_argument('-g', '--gpu', action='store_true',
                        help='train using GPU (default: False)')

    parser.add_argument('-d', '--dropout', default=0.30, type=float,
                        help='dropout value used during training (default: 0.30)')
    args = parser.parse_args()

    return args


def main():
    # so other functions can access these variables
    global args, dataloaders, data_sizes, image_sets
    args = get_user_args()

    # defining processing device, if cuda is available then GPU else CPU
    device = torch.device("cuda:0" if (
        torch.cuda.is_available() and args.gpu) else "cpu")

    print('=> beginning training using {}'.format(str(device).upper()))

    # lets the user know which model is being trained
    print('=> creating model: {}'.format(args.arch))
    model = models.__dict__[args.arch](pretrained=True)

    print('* ' * 20)

    model.to(device)  # send device to processor

    # image location with child folders of train, valid, test
    data_dir = Path(args.data)
    train_dir = data_dir / 'train'
    valid_dir = data_dir / 'valid'
    test_dir = data_dir / 'test'

    # variable for various iterations later
    states = ['train', 'valid', 'test']

    # for easy iteration later
    dirs_dict = {'train': train_dir, 'valid': valid_dir, 'test': test_dir}

    # image normalization parameters, predefined
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    # transforms for valid and test data, use same parameters
    valid_test_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize]

    data_transforms = {
        'train':  # vector manipulation for generalized learning
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'valid':
        transforms.Compose(valid_test_transforms),
        'test':
        transforms.Compose(valid_test_transforms)
    }

    image_sets = {
        i_set: datasets.ImageFolder(
            dirs_dict[i_set], transform=data_transforms[i_set])
        for i_set in states
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            image_sets['train'], batch_size=args.batch_size, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_sets['valid'],       batch_size=args.batch_size),
        'test': torch.utils.data.DataLoader(image_sets['test'], batch_size=args.batch_size)
    }
    classes = image_sets['train'].classes
    data_sizes = {x: len(image_sets[x]) for x in states}

    for p in model.parameters():
        p.requires_grad = False  # ensures gradients aren't calculated for parameters

    classifier = nn.Sequential(
        OrderedDict([
            ('fc1', nn.Linear(
                model.classifier[0].in_features, args.hidden_units)),
            ('relu1,', nn.ReLU()),
            ('dropout', nn.Dropout(args.dropout)),
            ('fc2', nn.Linear(args.hidden_units, len(classes))),
            ('output', nn.LogSoftmax(dim=1)),
        ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(),
                           lr=args.learning_rate)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.0125)

    model_trained = train(model, optimizer, criterion,
                          scheduler, args.epochs, device)

    save_checkpoint(model_trained, args.epochs, args.save_dir,
                    args.arch, args.learning_rate, optimizer, args.hidden_units)


def train(model, optimizer, criterion, scheduler, num_epochs, device):
    """
    trains a selected model using CNN
    Parameters:
        model - Model selected from list of available models
        optimizer - optimizer used during training
        criterion - loss function to optimize
        scheduler - learning rate scheduler
        num_epochs - number of epochs to train the model with (default: 5)
        device - gpu or cpu (default: cpu)
    Returns:
        The trained model
    """

    begin = time.time()  # global loop time

    top_acc = 0.
    top_model_weights = copy.deepcopy(model.state_dict())

    for e in range(num_epochs):
        epoch_start = time.time()  # epoch starting time
        print('=> running epoch {}/{}'.format(e + 1, num_epochs))

        for state in ['train', 'valid']:
            if state == 'train':
                scheduler.step()  # changes learning rate
                model.train()  # this puts the model into a training state
            else:
                model.eval()  # this puts the model into an evaluation state

            running_loss = 0.
            running_correct = 0

            for images, labels in dataloaders[state]:
                images = images.to(device)  # send to GPU or CPU
                labels = labels.to(device)  # send to GPU or CPU

                optimizer.zero_grad()

                with torch.set_grad_enabled(state == 'train'):
                    model = model.to(device)  # sends to GPU or CPU
                    output = model(images)
                    _, predicts = torch.max(output, 1)
                    loss = criterion(output, labels)

                    if state == 'train':
                        loss.backward()  # backpropogation
                        optimizer.step()

                running_loss += loss.item() * images.size(0)  # add to running loss

                # total of correct predictions
                running_correct += torch.sum(predicts == labels.data)

            e_loss = running_loss / data_sizes[state]  # epoch loss

            # epoch accuracy, double() used to change tensor data type
            e_acc = running_correct.double() / data_sizes[state]

            # print running loss and accuracy for train and validation sets
            print('=> {} loss: {:.4f} acc: {:.4f}'.format(state, e_loss, e_acc))

            if state == 'valid' and e_acc > top_acc:
                top_acc = e_acc  # sets the top accuracy if exceeds existing max

                # save model state at highest accuracy
                top_model_weights = copy.deepcopy(model.state_dict())

        epoch_time = time.time() - epoch_start  # total epoch run time
        print('=> epoch run time: {:.0f}m {:.0f}s '.format(
            epoch_time // 60, epoch_time % 60))
        print('* ' * 20)

    total_time = time.time() - begin  # total train run time
    print('=> completed in {:.0f}m {:.0f}s'.format(
        total_time // 60, total_time % 60))
    print('=> highest accuracy: {:.4f}'.format(top_acc))

    # load highest accuracy model and return it
    model.load_state_dict(top_model_weights)

    return model


def save_checkpoint(model, epochs, save_dir, arch, learning_rate, optimizer, hidden_units):
    """
    saves the trained and tested module by saving a checkpoint file to an optional directory
    params:
        model - tested and trained CNN
        epochs - number of epochs used to train
        save_dir - directory to save the checkpoint file(default - current wd)
        arch - pass string value of architecture used for loading
    returns:
        none - use module to save checkpoint
    """

    model.class_to_idx = image_sets['train'].class_to_idx

    file_name = 'checkpoint-' + \
        str(datetime.now().strftime("%Y-%m-%d-%H:%M")) + \
        '.pth'  # dynamic time file naming
    model.cpu()  # make sure it saves locally
    saved_model = {
        'arch': arch,  # save model architecture
        'classifier': model.classifier,  # save classifier structure
        'class_to_idx': model.class_to_idx,  # save class to index mapping
        'epochs': epochs,  # save num of epochs
        'hidden_units': hidden_units,  # save num of hidden units
        'learning_rate': learning_rate,  # save learning rate used
        'optimizer_dict': optimizer.state_dict(),  # save optimizer state dictionary
        'state_dict': model.state_dict(),  # save model state dict
    }

    if len(args.save_dir) == 0:
        save_path = save_dir + file_name  # save in current directory if no path provided
    else:
        save_path = save_dir + '/' + file_name  # save in specified file directory

    torch.save(saved_model, save_path)  # save file

    if len(args.save_dir) == 0:
        print('=> model checkpoint saved in current working directory')
    else:
        print('=> model checkpoint saved at {}'.format(save_path))


if __name__ == '__main__':
    main()
