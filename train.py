import torch
import numpy as np
import os
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn import metrics
from dataset import ESMH
from models import *
from utils import *
import torch.backends.cudnn as cudnn
import argparse

np.set_printoptions(precision=4)
cudnn.benchmark = True


def train(root_dir='/home/cvip/dataset/ESMH/cropped_patches/na_pd_2d/manual',
          test_dir=None,
          result_dir='/home/cvip/dataset/ESMH/subtype_results/curr_results',
          net='resnet', max_epochs=500, env_name='nnunet',
          fold=None, batch_size=32, num_classes=5, in_channels=3):

    plotter = VisdomLinePlotter(env_name=env_name)

    # data loader
    if fold is not None:
        data_path = os.path.join(root_dir, 'fold_' + str(fold))
    else:
        data_path = root_dir

    if test_dir is None:
        test_dir = data_path

    train_data_loader = DataLoader(dataset=ESMH(mode='train', data_path=data_path, dim='2d', use_phases=None, num_phases=in_channels), batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(dataset=ESMH(mode='test', data_path=test_dir, dim='2d', use_phases=None, num_phases=in_channels, label_path='/data/ESMH/subtypes.json'), batch_size=batch_size, shuffle=False)

    if fold is not None:
        result_dir = os.path.join(result_dir, 'fold_' + str(fold))

    result_dir = os.path.join(result_dir, net)

    os.makedirs(result_dir, exist_ok=True)

    if net == 'resnet':
        model = resnet_mod(in_channels, num_classes)
    elif net == 'googlenet':
        model = GoogleNet_mod(in_channels, num_classes)
    elif net == 'vgg':
        model = VGG_mod(in_channels, num_classes)
    elif net == 'vgg3d':
        model = VGG3D(in_channels, num_classes)
    else:
        print('wrong net name!')

    model.cuda()

    #weights = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    initial_lr = 1e-4
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=0.0002)
    # optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    for epoch in range(max_epochs):

        print('epoch: ', epoch)

        optimizer.param_groups[0]['lr'] = poly_lr(epoch, max_epochs, initial_lr)
        curr_lr = optimizer.param_groups[0]['lr']

        loss_avg = 0
        model.train()
        for batch_idx, data in enumerate(train_data_loader):
            image, label = data
            image, label = image.cuda(), label.cuda()

            optimizer.zero_grad()

            output = model(image)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            loss_avg += loss.item()

        loss_avg /= len(train_data_loader)
        print('train_loss_avg: {:.2f}, lr: {:.6f}'.format(loss_avg, curr_lr))
        plotter.plot('loss', 'train', 'Loss ' + result_dir.split('/')[-1], epoch, loss_avg)

        loss_avg = 0
        true_labels = np.array([])
        output_scores = np.array([]).reshape((0, num_classes))
        model.eval()
        for batch_idx, data in enumerate(val_data_loader):
            image, label = data
            image, label = image.cuda(), label.cuda()

            with torch.no_grad():
                output = model(image)

            loss = criterion(output, label)
            loss_avg += loss.item()

            output_sm = torch.nn.functional.softmax(output, dim=1)
            output_np = output_sm.detach().cpu().numpy()
            output_scores = np.append(output_scores, output_np, axis=0)
            true_labels = np.append(true_labels, label.cpu().numpy())

        auc = get_roc_auc(true_labels, output_scores, find_opt_thr=False, save_roc=True,
                          figure_name=os.path.join(result_dir, 'roc_val.jpg'))
        conf_mat = calc_conf_mat(true_labels, output_scores)
        sen, spe, acc = calc_eval_metric(conf_mat)

        loss_avg /= len(val_data_loader)
        print('test_loss_avg: {:.2f}, auc: {:.4f}, sen: {:.4f}, spe: {:.4f}, acc: {:.4f}, conf_mat: '.format(loss_avg, auc[5], sen[0], spe[0], acc[5]))
        print(conf_mat)
        plotter.plot('loss', 'test', 'Class Loss', epoch, loss_avg)
        plotter.plot('acc', 'auc', 'Test acc  ' + result_dir.split('/')[-2], epoch, auc[5])
        plotter.plot('acc', 'auc_0', 'Test acc  ' + result_dir.split('/')[-2], epoch, auc[0])
        plotter.plot('acc', 'auc_1', 'Test acc  ' + result_dir.split('/')[-2], epoch, auc[1])
        plotter.plot('acc', 'auc_2', 'Test acc  ' + result_dir.split('/')[-2], epoch, auc[2])
        plotter.plot('acc', 'auc_3', 'Test acc  ' + result_dir.split('/')[-2], epoch, auc[3])
        plotter.plot('acc', 'auc_4', 'Test acc  ' + result_dir.split('/')[-2], epoch, auc[4])
        # plotter.plot('acc', 'sen', 'Test acc  ' + result_dir.split('/')[-2], epoch, sen.mean())
        # plotter.plot('acc', 'spe', 'Test acc  ' + result_dir.split('/')[-2], epoch, spe.mean())
        # plotter.plot('acc', 'acc', 'Test acc  ' + result_dir.split('/')[-2], epoch, acc[5])

        if epoch % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(result_dir, 'model.pt'))

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(result_dir, 'model.pt'))

    print('done')


def main():
    # argument parser
    parser = argparse.ArgumentParser(description='ESMH')
    parser.add_argument('--root_dir', type=str, default='/data/ESMH/cropped_patches/p3_2d_est/split_13/fold_0')
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--result_dir', type=str, default='/data/ESMH/subtype_results/p3_2d_est/split_13/fold_0')
    parser.add_argument('--net', type=str, default='resnet')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--env_name', type=str, default='nnunet')
    args = parser.parse_args()
    print(args)

    train(fold=None, root_dir=args.root_dir, test_dir=args.test_dir, result_dir=args.result_dir, net=args.net, max_epochs=args.epochs, env_name=args.env_name)


if __name__ == "__main__":
    main()

