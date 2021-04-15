from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from sklearn.metrics import ConfusionMatrixDisplay
from dataset import ESMH
from models import *
from utils import *
import os
import argparse

np.set_printoptions(precision=4)


def eval_trial(root_dir='/home/cvip/dataset/ESMH/cropped_patches/na_pd_2d/split_50',
               result_dir='/home/cvip/dataset/ESMH/subtype_results/split_50',
               net='resnet', trial=None, num_classes=5, in_channels=3, mode='test', data_dim='2d', use_phases=None):
    if trial is not None:
        result_dir = os.path.join(result_dir, 'trial_' + str(trial), net)
        # result_dir = os.path.join(result_dir, 'trial_' + str(trial))

    test_data_loader = DataLoader(
        dataset=ESMH(mode='test', data_path=root_dir, dim='2d', use_phases=use_phases, num_phases=in_channels,
                     label_path='/data/ESMH/TCGA_subtypes.json'), batch_size=32, shuffle=False)

    if net == 'resnet':
        model = resnet_mod(in_channels, num_classes)
    elif net == 'resnet3d':
        model = resnet3d(in_channels, num_classes)
    elif net == 'densenet':
        model = denseNet_mod(in_channels, num_classes)
    elif net == 'googlenet':
        model = GoogleNet_mod(in_channels, num_classes)
    elif net == 'vgg':
        model = VGG_mod(in_channels, num_classes)
    elif net == 'vgg3d':
        model = VGG3D(in_channels, num_classes)
    else:
        print('wrong net name')
        return

    model.cuda()

    checkpoint = torch.load(os.path.join(result_dir, 'model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    true_labels = np.array([])
    output_scores = np.array([]).reshape((0, num_classes))
    model.eval()
    for batch_idx, data in enumerate(test_data_loader):
        image, label = data
        image, label = image.cuda(), label.cuda()

        with torch.no_grad():
            output = model(image)

        output_sm = torch.nn.functional.softmax(output, dim=1)
        output_np = output_sm.detach().cpu().numpy()
        output_scores = np.append(output_scores, output_np, axis=0)
        true_labels = np.append(true_labels, label.cpu().numpy())

    return true_labels, output_scores


def main():
    # argument parser
    parser = argparse.ArgumentParser(description='ESMH')
    parser.add_argument('--root_dir', type=str,
                        default='/data/ESMH/cropped_patches/tcga_3p_ref')
    parser.add_argument('--result_dir', type=str,
                        default='/data/ESMH/subtype_results/tcga_best')
    parser.add_argument('--net', type=str, default='resnet')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--env_name', type=str, default='nnunet')
    parser.add_argument('--data_dim', type=str, default='2d')
    parser.add_argument('--num_phase', type=int, default=2)
    args = parser.parse_args()

    scores_avg = np.zeros([])
    trial_list = [2, 3, 4, 7, 8, 9, 57, 58, 76, 84]

    for i in range(10):
        for j in range(3):
            if j == 0:
                phases = [0, 1, 2]
            elif j == 1:
                phases = [0, 1, 3]
            elif j == 2:
                phases = [0, 2, 3]
            else:
                break

            true_labels, scores = eval_trial(trial=trial_list[i], mode='test', net=args.net,
                                             root_dir=args.root_dir, result_dir=args.result_dir,
                                             data_dim=args.data_dim, use_phases=phases)

            scores_avg = scores_avg + scores

    scores_avg /= (10 * 3)

    auc, opt_thr = get_roc_auc_fig(true_labels, scores_avg, find_opt_thr=True, save_roc=True,
                                   figure_name=os.path.join(args.result_dir))
    print('auc: ', auc)
    print('opt_thr: ', opt_thr)

    auc = get_pr_auc_fig(true_labels, scores_avg, save_roc=True, figure_name=os.path.join(args.result_dir))

    conf_mat = calc_conf_mat(true_labels, scores_avg, thr=[0.07, 0.34, 0.44, 0.24, 0.38])
    conf_mat = conf_mat.astype(np.int)
    print(conf_mat)

    disp = ConfusionMatrixDisplay(conf_mat, ['ccRCC', 'pRCC', 'chRCC', 'AML', 'Oncocytoma'])
    disp.plot(figure_name=os.path.join(args.result_dir, 'conf_model.svg'))

    print("top-1 acc (conf):", calc_acc_from_conf_mat(conf_mat))
    print("top-1 acc:", calc_top1_acc_from_scores(true_labels, scores_avg))
    print("top-2 acc:", calc_top2_acc_from_scores(true_labels, scores_avg))

    sen, spe, acc = calc_eval_metric(conf_mat)
    print('sen: ', sen)
    print('spe: ', spe)
    print('acc: ', acc)

    print('main done')


if __name__ == "__main__":
    main()
