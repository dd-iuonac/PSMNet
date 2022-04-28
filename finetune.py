from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import copy
from dataloader import KITTILoader as kittiLoader

from models import *


def train(model, optimizer, imgL, imgR, disp_L, args):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
    else:
        disp_true = disp_L

    # ---------
    mask = (disp_true > 0)
    mask.detach_()
    # ----

    optimizer.zero_grad()

    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        loss = 0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7 * F.smooth_l1_loss(output2[mask], disp_true[mask],
                                                                                                                  size_average=True) + F.smooth_l1_loss(
            output3[mask], disp_true[mask], size_average=True)
    elif args.model == 'basic':
        output = model(imgL, imgR)
        output = torch.squeeze(output, 1)
        loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, imgL, imgR, disp_true, args):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    with torch.no_grad():
        output3 = model(imgL, imgR)
    output3.squeeze_()

    pred_disp = output3.data.cpu()

    # computing 3-px error#
    true_disp = copy.deepcopy(disp_true)
    index = np.argwhere(true_disp > 0)
    disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(
        true_disp[index[0][:], index[1][:], index[2][:]] - pred_disp[index[0][:], index[1][:], index[2][:]])
    correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3) | (
            disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]] * 0.05)
    torch.cuda.empty_cache()

    return 1 - (float(torch.sum(correct)) / float(len(index[0])))


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = 0.001
    else:
        lr = 0.0001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    parser = argparse.ArgumentParser(description='PSMNet')
    parser.add_argument('--maxdisp', type=int, default=192,
                        help='maxium disparity')
    parser.add_argument('--model', default='stackhourglass',
                        help='select model')
    parser.add_argument('--datatype', default='2015',
                        help='datapath')
    parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/training/',
                        help='datapath')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train')
    parser.add_argument('--train_batch_size', default=12, type=int, help="training batch size")
    parser.add_argument('--test_batch_size', default=8, type=int, help="testing batch size")
    parser.add_argument('--train_num_workers', default=8, type=int, help="training number of workers")
    parser.add_argument('--test_num_workers', default=4, type=int, help="testing number of workers")
    parser.add_argument('--loadmodel', default=None,
                        help='load model')
    parser.add_argument('--savemodel', default='./',
                        help='save model')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.datatype == '2015':
        from dataloader import KITTIloader2015 as dataLoader
    elif args.datatype == '2012':
        from dataloader import KITTIloader2012 as dataLoader

    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = dataLoader.dataloader(args.datapath)

    TrainImgLoader = torch.utils.data.DataLoader(
        kittiLoader.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
        batch_size=args.train_batch_size, shuffle=True, num_workers=args.train_num_workers, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        kittiLoader.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=args.test_batch_size, shuffle=False, num_workers=args.test_num_workers, drop_last=False)

    model = None
    if args.model == 'stackhourglass':
        model = stackhourglass(args.maxdisp, args.cuda)
    elif args.model == 'basic':
        model = basic(args.maxdisp)
    else:
        print('no model')
        exit(1)

    if args.cuda:
        model = nn.DataParallel(model)
        model.cuda()

    if args.loadmodel is not None:
        print(f"Loading model: {args.loadmodel}")
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'])
    else:
        print("Init new model")

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

    # Training and testing
    max_acc = 0
    max_epo = 0
    start_full_time = time.time()

    for epoch in range(1, args.epochs + 1):
        total_train_loss = 0
        total_test_loss = 0
        adjust_learning_rate(optimizer, epoch)

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()

            loss = train(imgL_crop, imgR_crop, disp_crop_L)
            print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)))

        ## Test ##

        for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            test_loss = test(imgL, imgR, disp_L)
            print('Iter %d 3-px error in val = %.3f' % (batch_idx, test_loss * 100))
            total_test_loss += test_loss

        print('epoch %d total 3-px error in val = %.3f' % (epoch, total_test_loss / len(TestImgLoader) * 100))
        if total_test_loss / len(TestImgLoader) * 100 > max_acc:
            max_acc = total_test_loss / len(TestImgLoader) * 100
            max_epo = epoch
        print('MAX epoch %d total test error = %.3f' % (max_epo, max_acc))

        # Save
        savefilename = args.savemodel + 'finetune_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss / len(TrainImgLoader),
            'test_loss': total_test_loss / len(TestImgLoader) * 100,
        }, savefilename)

    print('full finetune time = %.2f HR' % ((time.time() - start_full_time) / 3600))
    print(max_epo)
    print(max_acc)


if __name__ == '__main__':
    main()
