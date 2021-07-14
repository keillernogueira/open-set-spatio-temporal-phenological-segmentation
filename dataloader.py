import os
import imageio
import numpy as np

import torch
from torch.utils import data


class DataLoader(data.Dataset):

    def __init__(self, mode, images, mask, distr_data, patch_size):
        super().__init__()

        self.mode = mode
        self.images = images
        self.mask = mask
        self.distr_data = distr_data
        self.patch_size = patch_size

        self.num_classes = 4


    @staticmethod
    def manipulate_border_array(data, crop_size):
        mask = int(crop_size / 2)
        # print data.shape

        h, w = len(data), len(data[0])
        crop_left = data[0:h, 0:crop_size, :]
        crop_right = data[0:h, w - crop_size:w, :]
        crop_top = data[0:crop_size, 0:w, :]
        crop_bottom = data[h - crop_size:h, 0:w, :]

        mirror_left = np.fliplr(crop_left)
        mirror_right = np.fliplr(crop_right)
        flipped_top = np.flipud(crop_top)
        flipped_bottom = np.flipud(crop_bottom)

        h_new, w_new = h + mask * 2, w + mask * 2
        data_border = np.zeros((h_new, w_new, len(data[0][0])))
        # print data_border.shape
        data_border[mask:h + mask, mask:w + mask, :] = data

        data_border[mask:h + mask, 0:mask, :] = mirror_left[:, mask + 1:, :]
        data_border[mask:h + mask, w_new - mask:w_new, :] = mirror_right[:, 0:mask, :]
        data_border[0:mask, mask:w + mask, :] = flipped_top[mask + 1:, :, :]
        data_border[h + mask:h + mask + mask, mask:w + mask, :] = flipped_bottom[0:mask, :, :]

        data_border[0:mask, 0:mask, :] = flipped_top[mask + 1:, 0:mask, :]
        data_border[0:mask, w + mask:w + mask + mask, :] = flipped_top[mask + 1:, w - mask:w, :]
        data_border[h + mask:h + mask + mask, 0:mask, :] = flipped_bottom[0:mask, 0:mask, :]
        data_border[h + mask:h + mask + mask, w + mask:w + mask + mask, :] = flipped_bottom[0:mask, w - mask:w, :]

        # scipy.misc.imsave('C:\\Users\\Keiller\\Desktop\\outfile.jpg', data_border)
        return data_border

    @staticmethod
    def load_images(dataset_path, images, patch_size):
        data = []

        for name in images:
            try:
                img = imageio.imread(os.path.join(dataset_path, name + '.tif'))
                print(name, img.shape)
                # img_as_float(imageio.imread(os.path.join(dataset_path, name)))
            except IOError:
                raise IOError("Could not open file: ", os.path.join(dataset_path, name))

            data.append(DataLoader.manipulate_border_array(img, patch_size))

        try:
            mask = imageio.imread(os.path.join(dataset_path, "mask_train_test_int.png"))
        except IOError:
            raise IOError("Could not open file: " + os.path.join(dataset_path, "mask_train_test_int.png"))

        print(np.bincount(mask.flatten()))
        return np.asarray(data), np.asarray(mask)

    @staticmethod
    def create_distributions_over_pixel_classes(labels, num_classes):
        training_instances = [[[] for i in range(0)] for i in range(num_classes)]
        testing_instances = [[[] for i in range(0)] for i in range(num_classes)]
        no_classes_instances = []

        w, h = labels.shape

        for i in range(0, w):
            for j in range(0, h):
                if labels[i, j] != 8:
                    if labels[i, j] == 0 or labels[i, j] == 1 or labels[i, j] == 2 or labels[i, j] == 3:
                        training_instances[labels[i, j]].append((i, j))
                    else:
                        testing_instances[labels[i, j] - 4].append((i, j))
                else:
                    no_classes_instances.append((i, j))

        for i in range(len(training_instances)):
            print("Training class " + str(i) + " = " + str(len(training_instances[i])))
            print("Testing class " + str(i) + " = " + str(len(testing_instances[i])))
        print('No class = ' + str(len(no_classes_instances)))

        train_data = np.asarray(training_instances[0] + training_instances[1] +
                                training_instances[2] + training_instances[3])
        test_data = np.asarray(testing_instances[0] + testing_instances[1] +
                               testing_instances[2] + testing_instances[3])
        return train_data, test_data, np.asarray(no_classes_instances)

    @staticmethod
    def data_augmentation(img, msk=None, msk_true=None):
        rand_fliplr = np.random.random() > 0.50
        rand_flipud = np.random.random() > 0.50
        rand_rotate = np.random.random()

        if rand_fliplr:
            img = np.flip(img, axis=2)
            if msk is not None:
                msk = np.flip(msk, axis=2)
            if msk_true is not None:
                msk_true = np.flip(msk_true, axis=2)
        if rand_flipud:
            img = np.flip(img, axis=1)
            if msk is not None:
                msk = np.flip(msk, axis=1)
            if msk_true is not None:
                msk_true = np.flip(msk_true, axis=1)

        if rand_rotate < 0.25:
            img = np.rot90(img, 3, (1, 2))
            if msk is not None:
                msk = np.rot90(msk, 3, (1, 2))
            if msk_true is not None:
                msk_true = np.rot90(msk_true, 3, (1, 2))
        elif rand_rotate < 0.50:
            img = np.rot90(img, 2, (1, 2))
            if msk is not None:
                msk = np.rot90(msk, 2, (1, 2))
            if msk_true is not None:
                msk_true = np.rot90(msk_true, 2, (1, 2))
        elif rand_rotate < 0.75:
            img = np.rot90(img, 1, (1, 2))
            if msk is not None:
                msk = np.rot90(msk, 1, (1, 2))
            if msk_true is not None:
                msk_true = np.rot90(msk_true, 1, (1, 2))

        img = img.astype(np.float32)
        if msk is not None:
            msk = msk.astype(np.int64)
        if msk_true is not None:
            msk_true = msk_true.astype(np.int64)

        return img, msk, msk_true

    def __getitem__(self, index):
        mask = int(self.patch_size / 2)

        cur_x = self.distr_data[index][0]
        cur_y = self.distr_data[index][1]

        cur_path = self.images[:, (cur_x + mask) - mask:(cur_x + mask) + mask + 1,
                               (cur_y + mask) - mask:(cur_y + mask) + mask + 1, :]
        cur_mask = self.mask[cur_x, cur_y]
        # cur_bool_mask = np.array(np.array(cur_mask, dtype=bool), dtype=int)

        if cur_mask > 3:
            cur_mask = cur_mask - 4

        assert len(cur_path[0]) == self.patch_size and len(cur_path[0][0]) == self.patch_size, \
            "Wrong patch size " + str(len(cur_path[0])) + " x " + str(len(cur_path[0][0]))
        # assert cur_mask == 0 or cur_mask == 1 or cur_mask == 2 or cur_mask == 3, \
        #     "Current class is wrong: " + str(cur_mask)

        cur_path = (cur_path / 255) - 0.5
        if self.mode == 'Train':
            cur_path, _, _ = self.data_augmentation(cur_path)
        cur_path = np.transpose(cur_path, (0, 3, 1, 2))

        # Turning to tensors.
        cur_path = torch.from_numpy(cur_path.copy())
        # cur_mask = torch.from_numpy(cur_mask.copy())
        # mask = torch.from_numpy(mask.copy())

        return cur_path.float(), cur_mask.astype(np.long), np.array([cur_x, cur_y])

    def __len__(self):
        return len(self.distr_data)
