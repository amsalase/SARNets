# -*- coding: utf-8 -*-
import os, argparse, time, datetime, stat, shutil
import numpy as np
import torch
import random
import warnings
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.utils as vutils
from util.MF_dataset import MF_dataset
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from util.util import compute_results
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from pytorch_toolbelt import losses as L
from loss_hub.losses import DiceLoss,SoftCrossEntropyLoss
from model.U_Net import UNet
from torch.cuda.amp import autocast,GradScaler
from torch import nn
from pytorchtools import EarlyStopping
""
parser = argparse.ArgumentParser(description='Train with pytorch')
""
parser.add_argument('--model_name', '-m', type=str, default='UNet_SE_tl1')
parser.add_argument('--batch_size', '-b', type=int, default=5)
parser.add_argument('--lr_start', '-ls', type=float, default=0.04)
parser.add_argument('--seed', '-seed',default=3407, type=int,help='seed for initializing training. ')
parser.add_argument('--gpu', '-g', type=int, default=0)
""
parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
parser.add_argument('--epoch_max', '-em', type=int, default=150) 
parser.add_argument('--epoch_from', '-ef', type=int, default=0)
parser.add_argument('--num_workers', '-j', type=int, default=8)
parser.add_argument('--n_class', '-nc', type=int, default=12)
parser.add_argument('--data_dir', '-dr', type=str, default='/workspace/Dataset_UMA/')
parser.add_argument('--patience', '-pa', type=int, default=100)
parser.add_argument('--load', action='store_true', default=True)
args = parser.parse_args()
""
augmentation_methods = [
    RandomBrightness(bright_range=0.5, prob=0.9),
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0),
    RandomCropOut(crop_rate=0.2, prob=1.0)
]
scaler = GradScaler()

def train(epo, model, train_loader, optimizer):
    # to track the training loss as the model trains
    train_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    model.train()
    for it, (images, labels, names) in enumerate(train_loader):
        images = Variable(images).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)
        start_t = time.time()  # time.time() returns the current time
        optimizer.zero_grad()
        logits = model(images)
        DiceLoss_fn = DiceLoss(mode="multiclass")
        SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)
        criterion = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn,first_weight=0.5, second_weight=0.5).cuda()
        loss = criterion(logits,labels)
        loss.backward()
        optimizer.step()
        lr_this_epo = 0
        train_losses.append(loss.item())
        for param_group in optimizer.param_groups:
            lr_this_epo = param_group['lr']
        print('Train: %s, epo %s/%s, iter %s/%s, lr %.8f, %.2f img/sec, loss %.4f, time %s' \
              % (args.model_name, epo, args.epoch_max, it + 1, len(train_loader), lr_this_epo,
                 len(names) / (time.time() - start_t), float(loss),
                 datetime.datetime.now().replace(microsecond=0) - start_datetime))
        if accIter['train'] % 1 == 0:
            writer.add_scalar('Train/loss', loss, accIter['train'])
        view_figure = True  # note that I have not colorized the GT and predictions here
        if accIter['train'] % 500 == 0:
            if view_figure:
                input_rgb_images = vutils.make_grid(images[:, :3], nrow=8,
                                                    padding=10)  # can only display 3-channel images, so images[:,:3]
                writer.add_image('Train/input_rgb_images', input_rgb_images, accIter['train'])
                scale = max(1,
                            255 // args.n_class)  # label (0,1,2..) is invisable, multiply a constant for visualization
                groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor),
                                               1)  # change to 3-channel for visualization
                groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                writer.add_image('Train/groudtruth_images', groudtruth_images, accIter['train'])
                predicted_tensor = logits.argmax(1).unsqueeze(
                    1) * scale  # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor),
                                             1)  # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                writer.add_image('Train/predicted_images', predicted_images, accIter['train'])
        accIter['train'] = accIter['train'] + 1
    train_loss = np.average(train_losses)
    avg_train_losses.append(train_loss)
    return avg_train_losses


def validation(epo, model, val_loader):
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    model.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(val_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            start_t = time.time()  # time.time() returns the current time
            logits = model(images)
            DiceLoss_fn = DiceLoss(mode="multiclass")
            SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)
            criterion = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn,first_weight=0.5, second_weight=0.5).cuda()
            loss = criterion(logits,labels)
            valid_losses.append(loss.item())
            print('Val: %s, epo %s/%s, iter %s/%s, %.2f img/sec, loss %.4f, time %s' \
                  % (
                  args.model_name, epo, args.epoch_max, it + 1, len(val_loader), len(names) / (time.time() - start_t),float(loss),
                  datetime.datetime.now().replace(microsecond=0) - start_datetime))
            if accIter['val'] % 1 == 0:
                writer.add_scalar('Validation/loss', loss, accIter['val'])
            view_figure = False  # note that I have not colorized the GT and predictions here
            if accIter['val'] % 100 == 0:
                if view_figure:
                    input_rgb_images = vutils.make_grid(images[:, :3], nrow=8,
                                                        padding=10)  # can only display 3-channel images, so images[:,:3]
                    writer.add_image('Validation/input_rgb_images', input_rgb_images, accIter['val'])
                    scale = max(1,
                                255 // args.n_class)  # label (0,1,2..) is invisable, multiply a constant for visualization
                    groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                    groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor),
                                                   1)  # change to 3-channel for visualization
                    groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/groudtruth_images', groudtruth_images, accIter['val'])
                    predicted_tensor = logits.argmax(1).unsqueeze(
                        1) * scale  # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                    predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor),
                                                 1)  # change to 3-channel for visualization, mini_batch*1*480*640
                    predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/predicted_images', predicted_images, accIter['val'])
            accIter['val'] += 1
            valid_loss = np.average(valid_losses)
            avg_valid_losses.append(valid_loss)
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
                        
        return avg_valid_losses, valid_loss


def testing(epo, model, test_loader):
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    label_list = ["unlabeled", "first-responder", "civilian", "vegetation", "road", "dirt-road", "building", "sky", "civilian-car", "response-vehicle", "debris", "command-post"]
    testing_results_file = os.path.join(weight_dir, 'testing_results_file.txt')
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            logits = model(images)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(
                1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])  # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            print('Test: %s, epo %s/%s, iter %s/%s, time %s' % (
            args.model_name, epo, args.epoch_max, it + 1, len(test_loader),
            datetime.datetime.now().replace(microsecond=0) - start_datetime))
    precision, recall, IoU = compute_results(conf_total)
    writer.add_scalar('Test/average_precision', precision.mean(), epo)
    writer.add_scalar('Test/average_recall', recall.mean(), epo)
    writer.add_scalar('Test/average_IoU', IoU.mean(), epo)
    for i in range(len(precision)):
        writer.add_scalar("Test(class)/precision_class_%s" % label_list[i], precision[i], epo)
        writer.add_scalar("Test(class)/recall_class_%s" % label_list[i], recall[i], epo)
        writer.add_scalar('Test(class)/Iou_%s' % label_list[i], IoU[i], epo)
    if epo == 0:
        with open(testing_results_file, 'w') as f:
            f.write("# %s, initial lr: %s, batch size: %s, date: %s \n" % (
            args.model_name, args.lr_start, args.batch_size, datetime.date.today()))
            f.write(
                "# epoch: unlabeled, first-responder, civilian, vegetation, road, dirt-road, building, sky, civilian-car, response-vehicle, debris, command-post, average(nan_to_num). (Acc %, IoU %)\n")
    with open(testing_results_file, 'a') as f:
        f.write(str(epo) + ': ')
        for i in range(len(precision)):
            f.write('%0.4f, %0.4f, ' % (100 * recall[i], 100 * IoU[i]))
        f.write('%0.4f, %0.4f\n' % (100 * np.mean(np.nan_to_num(recall)), 100 * np.mean(np.nan_to_num(IoU))))
    print('saving testing results.')
    with open(testing_results_file, "r") as file:
        writer.add_text('testing_results', file.read().replace('\n', '  \n'), epo)
    return 100 * np.mean(np.nan_to_num(IoU)) #añadido para early stop

# +
if __name__ == '__main__':
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')
    model = UNet(args.n_class)
    if args.gpu >= 0: model.cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=-1)

    weight_dir = os.path.join("./2da_etapa/UNet_SE_tl1_weight/", args.model_name)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    os.chmod(weight_dir,
             stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine

    writer = SummaryWriter("./2da_etapa/UNet_SE_tl1_weight/tensorboard_log")
    os.chmod("./2da_etapa/UNet_SE_tl1_weight/tensorboard_log",
             stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine
    os.chmod("./2da_etapa/UNet_SE_tl1_weight", stat.S_IRWXO)

    print('training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('weight will be saved in: %s' % weight_dir)

    train_dataset = MF_dataset(data_dir=args.data_dir, split='train', transform=augmentation_methods)
    val_dataset = MF_dataset(data_dir=args.data_dir, split='val')
    test_dataset = MF_dataset(data_dir=args.data_dir, split='test')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    start_datetime = datetime.datetime.now().replace(microsecond=0)
    accIter = {'train': 0, 'val': 0}
    # initialize the early_stopping object
    patience = args.patience
    if args.load:
        print("adentro")
        model.load_state_dict(torch.load('checkpoint_UNet_SE.pt'))
    early_stopping = EarlyStopping(patience=patience, verbose=True,path='./2da_etapa/checkpoint_UNet_SE_tl1.pt')
    for epo in range(args.epoch_from, args.epoch_max):
        print('\ntrain %s, epo #%s begin...' % (args.model_name, epo))
        avg_train_losses = train(epo, model, train_loader, optimizer)
        avg_valid_losses, valid_loss = validation(epo, model, val_loader)
        #testing(epo, model, test_loader)
        miou = testing(epo, model, test_loader) #añadido para earlystop
        early_stopping(valid_loss, miou, model)
        #early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        scheduler.step()
        
        
