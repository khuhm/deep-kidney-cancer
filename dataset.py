from torch.utils.data import Dataset
import random
import torch.nn.functional as F
from batchgenerators.augmentations.spatial_transformations import augment_mirroring, augment_spatial, augment_rot90
from batchgenerators.augmentations.noise_augmentations import augment_gaussian_noise, augment_gaussian_blur
from utils import *


class ESMH(Dataset):
    def __init__(self, mode='train', data_path='/data/ESMH/cropped_patches/na_pd_2d/fold_0',
                 label_path='/data/ESMH/subtypes.json', dim='2d', use_phases=None, num_phases=3):
        self.mode = mode
        self.data_path = data_path
        self.label_path = label_path
        self.dim = dim
        self.use_phases = use_phases
        self.num_phases = num_phases

        files = os.listdir(os.path.join(data_path, mode, 'image'))
        files = sorted(files)
        self.case_list = [case for case in files if case[-8] == '0']

        with open(label_path) as json_file:
            self.subtype_list = json.load(json_file)

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        case = self.case_list[idx]
        label = self.subtype_list[case[:-4]]

        case_n = os.path.join(self.data_path, self.mode, 'image', case)
        case_a = os.path.join(self.data_path, self.mode, 'image', 'case_0' + '1' + case[7:])
        case_p = os.path.join(self.data_path, self.mode, 'image', 'case_0' + '2' + case[7:])
        case_d = os.path.join(self.data_path, self.mode, 'image', 'case_0' + '3' + case[7:])

        seg_path_n = os.path.join(self.data_path, self.mode, 'seg', case)
        seg_path_a = os.path.join(self.data_path, self.mode, 'seg', 'case_0' + '1' + case[7:])
        seg_path_p = os.path.join(self.data_path, self.mode, 'seg', 'case_0' + '2' + case[7:])
        seg_path_d = os.path.join(self.data_path, self.mode, 'seg', 'case_0' + '3' + case[7:])

        img_list = []
        phase_list = []

        if os.path.isfile(case_n):
            img_n = np.load(case_n)
            seg_n = np.load(seg_path_n)
            img_n = torch.from_numpy(img_n).to(torch.float32)
            img_list.append(img_n)
            phase_list.append(0)

        if os.path.isfile(case_a):
            img_a = np.load(case_a)
            seg_a = np.load(seg_path_a)
            img_a = torch.from_numpy(img_a).to(torch.float32)
            img_list.append(img_a)
            phase_list.append(1)
        else:
            img_a = torch.zeros_like(img_n)

        if os.path.isfile(case_p):
            img_p = np.load(case_p)
            seg_p = np.load(seg_path_p)
            img_p = torch.from_numpy(img_p).to(torch.float32)
            img_list.append(img_p)
            phase_list.append(2)
        else:
            img_p = torch.zeros_like(img_n)

        if os.path.isfile(case_d):
            img_d = np.load(case_d)
            seg_d = np.load(seg_path_d)
            img_d = torch.from_numpy(img_d).to(torch.float32)
            img_list.append(img_d)
            phase_list.append(3)
        else:
            img_d = torch.zeros_like(img_n)

        if self.dim == '2d':
            target_size = 224

            for i, img in enumerate(img_list):
                img_list[i] = F.interpolate(img.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)

        elif self.dim == '3d':
            target_size = (16, 64, 64)

            img_n = torch.from_numpy(img_n).to(torch.float32)
            img_p = torch.from_numpy(img_p).to(torch.float32)
            img_d = torch.from_numpy(img_d).to(torch.float32)

            img_n = F.interpolate(img_n.unsqueeze(0).unsqueeze(0), size=target_size, mode='trilinear', align_corners=False).squeeze(0)
            img_p = F.interpolate(img_p.unsqueeze(0).unsqueeze(0), size=target_size, mode='trilinear', align_corners=False).squeeze(0)
            img_d = F.interpolate(img_d.unsqueeze(0).unsqueeze(0), size=target_size, mode='trilinear', align_corners=False).squeeze(0)

            image = torch.cat((img_n, img_p, img_d))
        else:
            print('dim error')

        if len(img_list) > self.num_phases:
            if self.use_phases is None:
                use_phases = [0] + sorted(random.sample(list(range(1, len(img_list))), self.num_phases - 1))
                # use_phases = sorted(random.sample(list(range(0, len(img_list))), self.num_phases))
                temp_list = [img for i, img in enumerate(img_list) if i in use_phases]
            else:
                temp_list = [img for i, img in enumerate(img_list) if phase_list[i] in self.use_phases]
                if len(temp_list) < self.num_phases:
                    use_phases = [0] + sorted(random.sample(list(range(1, len(img_list))), self.num_phases - 1))
                    # use_phases = sorted(random.sample(list(range(0, len(img_list))), self.num_phases))
                    temp_list = [img for i, img in enumerate(img_list) if i in use_phases]
            image = torch.cat(temp_list)
        else:
            image = torch.cat(img_list)

        if self.mode != 'train':
            return image, label

        image = image.numpy()

        image, _ = augment_mirroring(image)

        image, _ = augment_rot90(image, sample_seg=None, num_rot=(0, 1, 2, 3), axes=(0, 1))

        if np.random.uniform() < 0.1: # 0.1
            image = augment_gaussian_noise(image)

        if np.random.uniform() < 0.2: # 0.2
            image = augment_gaussian_blur(image, (0.5, 1.), p_per_channel=0.5)

        image = torch.from_numpy(image.copy()).to(torch.float32)

        return image, label