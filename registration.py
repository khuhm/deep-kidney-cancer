import torch
import numpy as np
import os
from torch.nn.functional import pad, one_hot, affine_grid, grid_sample
import torch.nn as nn


class AlignNet(torch.nn.Module):
    def __init__(self, init_trans=torch.zeros(3, 1), dof_6=False):
        super(AlignNet, self).__init__()
        self.angle = torch.nn.Parameter(torch.zeros(3))
        self.trans = torch.nn.Parameter(init_trans)
        theta = torch.cat((torch.eye(3), init_trans.unsqueeze(1)), dim=1).unsqueeze(0)
        self.theta = torch.nn.Parameter(theta)
        self.dof_6 = dof_6

    def forward(self, x):
        if not self.dof_6:
            grid = affine_grid(self.theta, x.size())
            out = grid_sample(x, grid, padding_mode='border')
            return out

    def transform(self, x):
        with torch.no_grad():
            r_t = torch.inverse(self.theta[0, :, :3])
            t_ = -torch.matmul(r_t, self.theta[0, :, 3:])
            out = torch.matmul(r_t, x) + t_
        return out


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        '''
        if shp_x[1] == 1:
            shp_x = torch.Size((shp_x[0], 3, shp_x[2], shp_x[3], shp_x[4]))
            net_output_ = net_output.long()
            net_output = torch.zeros(shp_x)
            if net_output_.device.type == "cuda":
                net_output = net_output.cuda(net_output.device.index)
            net_output.scatter_(1, net_output_, 1)
        '''
        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=False, smooth=1., tumor_only=False,
                 kidney_only=False):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.kidney_only = kidney_only
        self.tumor_only = tumor_only

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                if self.kidney_only:
                    dc = dc[:, 2]
                elif self.tumor_only:
                    dc = dc[:, 1]
                else:
                    dc = dc[:, 1:]

        return dc


def register(input_files, img_list, seg_list):
    """
        input_files: list of ct file path
        e.g)
        ['A': 'C:/PythonProjects/Kidney/3DProgram/data/image/case_00001\\image_A.nii.gz', 'C:/PythonProjects/Kidney/3DProgram/data/image/case_00001\\image_P.nii.gz', 'C:/PythonProjects/Kidney/3DProgram/data/image/case_00001\\image_D.nii.gz']

        img_list: list of ct image array
        seg_list: list of segmentation array
    """

    print(f'registration begin...')
    phase_list = [os.path.basename(input_file)[6] for input_file in input_files]

    if len(phase_list) == 1:
        print('1-phase', phase_list)
        return True

    # reference phase
    if 'P' in phase_list:
        ref_phase = 'P'
    elif 'A' in phase_list:
        ref_phase = 'A'
    elif 'D' in phase_list:
        ref_phase = 'D'
    else:
        ref_phase = None
        print('A, P, D all missing!')
        return False

    trans_mov_seg_data_list = []
    trans_mov_img_data_list = []

    trans_end_pt_list = []
    for i, phase in enumerate(phase_list):
        if phase == ref_phase:
            trans_mov_seg_data_list.append(None)
            trans_mov_img_data_list.append(None)
            continue

        ref_img_data = img_list[phase_list.index(ref_phase)]
        ref_seg_data = seg_list[phase_list.index(ref_phase)]

        mov_img_data = img_list[i]
        mov_seg_data = seg_list[i]

        mov_img_data = torch.from_numpy(mov_img_data)

        mov_seg_data = torch.from_numpy(mov_seg_data)
        ref_seg_data = torch.from_numpy(ref_seg_data)

        # 255 -> 0
        mov_seg_data[mov_seg_data == 255] = 0
        ref_seg_data[ref_seg_data == 255] = 0

        max_size = [max(mov_seg_data.size(0), ref_seg_data.size(0)),
                    max(mov_seg_data.size(1), ref_seg_data.size(1)),
                    max(mov_seg_data.size(2), ref_seg_data.size(2))]

        pad_mov_needed = [max_size[0] - mov_seg_data.size(0),
                          max_size[1] - mov_seg_data.size(1),
                          max_size[2] - mov_seg_data.size(2)]
        pad_mov = [int(pad_mov_needed[2] / 2), pad_mov_needed[2] - int(pad_mov_needed[2] / 2),
                   int(pad_mov_needed[1] / 2), pad_mov_needed[1] - int(pad_mov_needed[1] / 2),
                   int(pad_mov_needed[0] / 2), pad_mov_needed[0] - int(pad_mov_needed[0] / 2)]

        pad_ref_needed = [max_size[0] - ref_seg_data.size(0),
                          max_size[1] - ref_seg_data.size(1),
                          max_size[2] - ref_seg_data.size(2)]

        ref_margin = [int(pad_ref_needed[2] / 2), pad_ref_needed[2] - int(pad_ref_needed[2] / 2),
                      int(pad_ref_needed[1] / 2), pad_ref_needed[1] - int(pad_ref_needed[1] / 2),
                      int(pad_ref_needed[0] / 2), pad_ref_needed[0] - int(pad_ref_needed[0] / 2)]

        pad_ref = [0, 0,
                   0, 0,
                   0, pad_ref_needed[0]]

        slicer = [(ref_margin[0], ref_margin[0] + ref_seg_data.size(2) - 1),
                  (ref_margin[2], ref_margin[2] + ref_seg_data.size(1) - 1)]

        end_pt = torch.tensor([[pad_mov[4], pad_mov[4] + mov_seg_data.size(0) - 1],
                               [ref_seg_data.size(1) / 2, ref_seg_data.size(1) / 2],
                               [ref_seg_data.size(2) / 2, ref_seg_data.size(2) / 2]], dtype=torch.float)

        mov_seg_data = pad(mov_seg_data, pad_mov)
        ref_seg_data = pad(ref_seg_data, pad_ref)
        mov_img_data = pad(mov_img_data.unsqueeze(0).unsqueeze(0), pad_mov, mode='replicate')

        mov_seg_data = mov_seg_data[:, slicer[0][0]: slicer[0][1] + 1, slicer[1][0]: slicer[1][1] + 1]
        mov_img_data = mov_img_data[:, :, :, slicer[0][0]: slicer[0][1] + 1, slicer[1][0]: slicer[1][1] + 1]

        end_pt = 2 * end_pt / (torch.tensor(ref_seg_data.size(), dtype=torch.float).unsqueeze(1) - 1)
        end_pt = end_pt - 1
        end_pt = end_pt.flip(dims=(0,))

        # difference of center of mass
        center_diff = torch.mean(torch.stack(torch.where(mov_seg_data > 0)).to(torch.float), dim=1) - torch.mean(
            torch.stack(torch.where(ref_seg_data > 0)).to(torch.float), dim=1)
        center_diff = 2 * center_diff / (torch.tensor(ref_seg_data.size(), dtype=torch.float) - 1)
        center_diff = center_diff.flip(dims=(0,))

        # device
        device = 'cuda:0'  # cpu
        mov_seg_data = mov_seg_data.unsqueeze(0).to(device)
        ref_seg_data = ref_seg_data.unsqueeze(0).to(device)
        end_pt = end_pt.to(dtype=torch.float, device=device)

        mov_img_data = mov_img_data.to(dtype=torch.float, device=device)

        mov_seg_data = one_hot(mov_seg_data.to(torch.long), num_classes=3).to(dtype=torch.float)
        mov_seg_data = mov_seg_data.permute(0, 4, 1, 2, 3)
        mov_seg_data[:, 2, :, :, :] = mov_seg_data[:, 1, :, :, :] + mov_seg_data[:, 2, :, :, :]

        ref_seg_data = one_hot(ref_seg_data.to(torch.long), num_classes=3).to(dtype=torch.float)
        ref_seg_data = ref_seg_data.permute(0, 4, 1, 2, 3)
        ref_seg_data[:, 2, :, :, :] = ref_seg_data[:, 1, :, :, :] + ref_seg_data[:, 2, :, :, :]

        # affine transformation model
        model = AlignNet(center_diff)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        dice_loss_fn = SoftDiceLoss()

        for curr_iter in range(200):
            optimizer.zero_grad()
            trans_mov_seg_data = model(mov_seg_data)
            dice_fg = dice_loss_fn(trans_mov_seg_data, ref_seg_data)
            loss = -torch.mean(dice_fg[:, :])
            loss.backward()

            if curr_iter == 199:
                break
            optimizer.step()

        # end point
        trans_end_pt = model.transform(end_pt).cpu()
        trans_end_pt = trans_end_pt.flip(dims=(0,))
        trans_end_pt = (1 + trans_end_pt) / 2 * (
                    torch.tensor(ref_seg_data.size()[2:], dtype=torch.float).unsqueeze(1) - 1)
        trans_end_pt_list.append((torch.ceil(trans_end_pt[0, 0]).item(), torch.floor(trans_end_pt[0, 1]).item()))

        # to array
        trans_mov_seg_data[:, 2, :, :, :] = trans_mov_seg_data[:, 2, :, :, :] - trans_mov_seg_data[:, 1, :, :, :]
        trans_mov_seg_data = torch.argmax(trans_mov_seg_data[0], dim=0)
        trans_mov_seg_data = np.array(trans_mov_seg_data.cpu())

        # move image
        trans_mov_img_data = model(mov_img_data).squeeze()
        trans_mov_img_data = np.array(trans_mov_img_data.detach().cpu())

        trans_mov_seg_data_list.append(trans_mov_seg_data)
        trans_mov_img_data_list.append(trans_mov_img_data)

    # overlapped slice range
    if trans_end_pt_list == []:
        ref_img_data = img_list[phase_list.index(ref_phase)]
        slice_range = (0, ref_img_data.shape[0] - 1)
        ref_seg_data = seg_list[phase_list.index(ref_phase)]
    else:
        trans_end_pt_arr = np.array(trans_end_pt_list)
        slice_range = max(np.max(trans_end_pt_arr[:, 0]), 0), min(np.min(trans_end_pt_arr[:, 1]),
                                                                  ref_img_data.shape[0] - 1)

    img_list[phase_list.index(ref_phase)] = np.transpose(ref_img_data[int(slice_range[0]):int(slice_range[1]) + 1])

    ref_seg_data = seg_list[phase_list.index(ref_phase)]
    seg_list[phase_list.index(ref_phase)] = np.transpose(ref_seg_data[int(slice_range[0]):int(slice_range[1]) + 1])

    for i, phase in enumerate(phase_list):
        if phase == ref_phase:
            continue

        trans_mov_seg_data = trans_mov_seg_data_list[i]
        trans_mov_img_data = trans_mov_img_data_list[i]

        seg_list[phase_list.index(phase)] = np.transpose(
            trans_mov_seg_data[int(slice_range[0]):int(slice_range[1]) + 1])

        img_list[phase_list.index(phase)] = np.transpose(
            trans_mov_img_data[int(slice_range[0]):int(slice_range[1]) + 1])

    print(f'registration done')
    return True


def crop_patch(image, seg):
    """
        image: CT image array
        seg: segmentation mask array
        output: largest tumor region cropped patch
    """
    seg_tumor = (seg == 1).astype(np.float32)
    seg_tumor = np.reshape(seg_tumor, (seg_tumor.shape[0], -1))
    seg_tumor = np.sum(seg_tumor, axis=1)
    slice_large_tumor = np.argmax(seg_tumor)
    image = image[(slice_large_tumor,), :]
    seg = seg[(slice_large_tumor,), :]

    tumor_idx = np.argwhere(seg == 1)
    min_point = np.min(tumor_idx, axis=0)
    max_point = np.max(tumor_idx, axis=0)

    image = image[min_point[0]:max_point[0] + 1, min_point[1]:max_point[1] + 1, min_point[2]:max_point[2] + 1]
    seg = seg[min_point[0]:max_point[0] + 1, min_point[1]:max_point[1] + 1, min_point[2]:max_point[2] + 1]
    return image, seg
