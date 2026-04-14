import os
import imageio
import numpy as np

import torch
from torch.utils import data


def load_images(dataset_path, images, patch_size, mask_name="mask_train_test_int.png"):
    data = []

    for name in images:
        if 'itirapina' in mask_name and int(os.path.basename(name).split("_")[1]) in [251, 255, 276, 278]:
            continue
        try:
            img = imageio.v2.imread(os.path.join(dataset_path, name))
            print(name, img.shape)
            # img_as_float(imageio.imread(os.path.join(dataset_path, name)))
        except IOError:
            raise IOError("Could not open file: ", os.path.join(dataset_path, name))

        if patch_size is not None:
            data.append(np.pad(img, pad_width=((patch_size//2, patch_size//2), (patch_size//2, patch_size//2), (0, 0)),
                               mode='reflect'))
        else:
            data.append(img)
    try:
        mask = imageio.v2.imread(os.path.join(dataset_path, mask_name))
    except IOError:
        raise IOError("Could not open file: " + os.path.join(dataset_path, mask_name))

    # check inf and NaN
    # print("check inf", np.isinf(mask.flatten()).any())
    # print("check NaN", np.isnan(mask.flatten()).any())
    return np.asarray(data), np.asarray(mask)


def create_distributions(labels, num_classes):
    training_instances = [[[] for i in range(0)] for i in range(num_classes)]
    testing_instances = [[[] for i in range(0)] for i in range(num_classes)]
    no_classes_instances = []

    w, h = labels.shape

    for i in range(0, w):
        for j in range(0, h):
            if labels[i, j] != num_classes * 2:  # *2 because of the train/test split, e.g., 0,1,2,3 for train and 4,5,6,7 for test
                if labels[i, j] < num_classes:
                    training_instances[labels[i, j]].append((i, j))
                else:
                    testing_instances[labels[i, j] - num_classes].append((i, j))
            else:
                no_classes_instances.append((i, j))

    for i in range(len(training_instances)):
        print("Training class " + str(i) + " = " + str(len(training_instances[i])))
        print("Testing class " + str(i) + " = " + str(len(testing_instances[i])))
    print('No class = ' + str(len(no_classes_instances)))

    train_data = np.asarray(sum(training_instances, []))
    test_data = np.asarray(sum(testing_instances, []))
    return train_data, test_data, np.asarray(no_classes_instances)


def create_patch_distributions(labels, patch_size, hidden_class, num_classes):
    training_instances = []
    testing_instances = []
    openset_instances = []

    w, h = labels.shape
    stride = patch_size // 2
    for i in range(0, w, stride):
        for j in range(0, h, stride):
            cur_x = i
            cur_y = j
            patch_class = labels[cur_x:cur_x + patch_size, cur_y:cur_y + patch_size]

            # check size, if it is not okay, adjust accordingly
            if len(patch_class) != patch_size and len(patch_class[0]) != patch_size:
                cur_x = cur_x - (patch_size - len(patch_class))
                cur_y = cur_y - (patch_size - len(patch_class[0]))
                patch_class = labels[cur_x:cur_x + patch_size, cur_y:cur_y + patch_size]
            elif len(patch_class) != patch_size:
                cur_x = cur_x - (patch_size - len(patch_class))
                patch_class = labels[cur_x:cur_x + patch_size, cur_y:cur_y + patch_size]
            elif len(patch_class[0]) != patch_size:
                cur_y = cur_y - (patch_size - len(patch_class[0]))
                patch_class = labels[cur_x:cur_x + patch_size, cur_y:cur_y + patch_size]
            # double check
            assert patch_class.shape == (patch_size, patch_size), "Error create_distributions_over_classes: " \
                                                                  "Current patch size is " + str(len(patch_class)) + \
                                                                  "x" + str(len(patch_class[0]))

            count = np.bincount(patch_class.astype(int).flatten(), minlength=9)
            train_sum = np.sum([x for x in count[0:num_classes] if x != hidden_class])
            test_sum = np.sum([x for x in count[num_classes:2*num_classes] if x != hidden_class+num_classes])
            openset_sum = count[hidden_class] + count[hidden_class+num_classes]
            if openset_sum > 0:
                # print('openset', count, hidden_class, train_sum, test_sum, openset_sum, cur_x, cur_y)
                openset_instances.append((cur_x, cur_y))
            elif train_sum > test_sum and train_sum > openset_sum:
                # print('train', count, hidden_class, train_sum, test_sum, openset_sum, cur_x, cur_y)
                training_instances.append((cur_x, cur_y))
            elif test_sum > 0:
                # print('test', count, hidden_class, train_sum, test_sum, openset_sum, cur_x, cur_y)
                testing_instances.append((cur_x, cur_y))

    return np.asarray(training_instances), np.asarray(testing_instances), np.asarray(openset_instances)


def data_augmentation(img, msk=None, msk_true=None):
    rand_fliplr = np.random.random() > 0.50
    rand_flipud = np.random.random() > 0.50
    rand_rotate = np.random.random()

    if rand_fliplr:
        img = np.flip(img, axis=2)
        if msk is not None:
            msk = np.flip(msk, axis=1)
        if msk_true is not None:
            msk_true = np.flip(msk_true, axis=1)
    if rand_flipud:
        img = np.flip(img, axis=1)
        if msk is not None:
            msk = np.flip(msk, axis=0)
        if msk_true is not None:
            msk_true = np.flip(msk_true, axis=0)

    if rand_rotate < 0.25:
        img = np.rot90(img, 3, (1, 2))
        if msk is not None:
            msk = np.rot90(msk, 3, (0, 1))
        if msk_true is not None:
            msk_true = np.rot90(msk_true, 3, (0, 1))
    elif rand_rotate < 0.50:
        img = np.rot90(img, 2, (1, 2))
        if msk is not None:
            msk = np.rot90(msk, 2, (0, 1))
        if msk_true is not None:
            msk_true = np.rot90(msk_true, 2, (0, 1))
    elif rand_rotate < 0.75:
        img = np.rot90(img, 1, (1, 2))
        if msk is not None:
            msk = np.rot90(msk, 1, (0, 1))
        if msk_true is not None:
            msk_true = np.rot90(msk_true, 1, (0, 1))

    img = img.astype(np.float32)
    if msk is not None:
        msk = msk.astype(np.int64)
    if msk_true is not None:
        msk_true = msk_true.astype(np.int64)

    return img, msk, msk_true


# Serra do Cipo dataset
class DataLoader(data.Dataset):
    def __init__(self, mode, images, mask, distr_data, patch_size):
        super().__init__()

        self.mode = mode
        self.images = images
        self.mask = mask
        self.distr_data = distr_data
        self.patch_size = patch_size

        self.num_classes = 4

    def __getitem__(self, index):
        mask = self.patch_size // 2

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
        if self.mode == 'train':
            cur_path, _, _ = data_augmentation(cur_path)
        cur_path = np.transpose(cur_path, (0, 3, 1, 2))

        # Turning to tensors.
        cur_path = torch.from_numpy(cur_path.copy())
        # cur_mask = torch.from_numpy(cur_mask.copy())
        # mask = torch.from_numpy(mask.copy())

        return cur_path.float(), cur_mask.astype(np.long), np.array([cur_x, cur_y])

    def __len__(self):
        return len(self.distr_data)


# Serra do Cipo dataset
class PatchDataLoader(data.Dataset):
    def __init__(self, mode, images, mask, distr_data, patch_size, num_classes, hidden_class):
        super().__init__()
        assert mode in ['train', 'test', 'open']

        self.mode = mode
        self.images = images
        self.mask = mask
        self.distr_data = distr_data
        self.patch_size = patch_size
        self.hidden_class = hidden_class

        self.ignore_index = num_classes
        self.open_set_class = num_classes + 1
        self.num_classes = num_classes

        self.train_mask, self.test_mask, self.open_set_mask = self.shift_mask()
        print('original mask', np.bincount(self.mask.flatten()))
        print('train mask', np.bincount(self.train_mask.flatten()))
        print('test mask', np.bincount(self.test_mask.flatten()))
        print('open set mask', np.bincount(self.open_set_mask.flatten()))

    def shift_mask(self):
        train_mask = np.copy(self.mask)
        test_mask = np.copy(self.mask)
        open_set_mask = np.copy(self.mask)

        # train mask: 0,1,2,3 => 0,1,2,3 (except hidden class) and 4 => ignore_index
        train_mask[train_mask == self.hidden_class] = self.ignore_index
        for i in range(self.hidden_class + 1, self.num_classes):
            train_mask[train_mask == i] = i - 1
        train_mask[train_mask >= self.num_classes] = self.ignore_index  # "background" will be ignored during training

        # test mask: 4,5,6,7 => 0,1,2,3 (except hidden class) and 0,1,2,3 => ignore_index
        for c in range(self.num_classes):
            test_mask[test_mask == c] = 99  # temporary flag to avoid confusion
        test_mask = test_mask - self.num_classes  # 4,5,6,7 => 0,1,2,3
        test_mask[test_mask == self.hidden_class] = self.ignore_index
        for i in range(self.hidden_class+1, self.num_classes):
            test_mask[test_mask == i] = i - 1
        test_mask[test_mask >= self.num_classes] = self.ignore_index  # "background" will be ignored during training

        # open set mask: 4,5,6,7 => 0,1,2,3 , open set value = self.open_set_class
        open_set_mask[open_set_mask == self.hidden_class] = self.open_set_class + self.num_classes  # because of the -4 below
        open_set_mask[open_set_mask == self.hidden_class + self.num_classes] = self.open_set_class + self.num_classes  # because of the -4 below
        for c in range(self.num_classes):
            open_set_mask[open_set_mask == c] = 99  # temporary flag to avoid confusion
        open_set_mask = open_set_mask - self.num_classes  # 4,5,6,7 => 0,1,2,3  # only interested in test data
        for i in range(self.hidden_class + 1, self.num_classes):
            open_set_mask[open_set_mask == i] = i - 1
        open_set_mask[open_set_mask == (99-self.num_classes)] = self.ignore_index

        return train_mask, test_mask, open_set_mask

    # @deprecated
    def shift_patch_mask(self, cur_mask):
        openset_mask = np.copy(cur_mask)

        if self.mode == 'test':
            # +4 to contrapose the -4 below, in the end, this will be 8 and will be ignored
            for c in range(self.num_classes):
                cur_mask[cur_mask == c] = self.ignore_index + 4
            cur_mask = cur_mask - 4  # 4,5,6,7 => 0,1,2,3
        cur_mask[cur_mask == self.hidden_class] = self.ignore_index
        for i in range(self.hidden_class+1, self.num_classes):
            cur_mask[cur_mask == i] = i - 1
        cur_mask[cur_mask >= 4] = self.ignore_index  # class=8 is "background" and will be ignored during training

        openset_mask[openset_mask == self.hidden_class] = 15
        openset_mask[openset_mask == self.hidden_class + 4] = 15
        for c in range(self.num_classes):
            openset_mask[openset_mask == c] = self.ignore_index + 4
        openset_mask = openset_mask - 4  # 4,5,6,7 => 0,1,2,3  # only interested in test data
        for i in range(self.hidden_class + 1, self.num_classes):
            openset_mask[openset_mask == i] = i - 1
        openset_mask[openset_mask == 15 - 4] = self.num_classes - 1
        openset_mask[openset_mask >= 4] = self.ignore_index

        return cur_mask, openset_mask

    def __getitem__(self, index):
        cur_x = self.distr_data[index][0]
        cur_y = self.distr_data[index][1]

        cur_path = self.images[:, cur_x:cur_x + self.patch_size, cur_y:cur_y + self.patch_size, :]
        if self.mode == 'train':
            cur_mask = np.copy(self.train_mask[cur_x:cur_x + self.patch_size, cur_y:cur_y + self.patch_size])
            # print('data validation', np.bincount(cur_mask.flatten()))
        elif self.mode == 'test':
            cur_mask = np.copy(self.test_mask[cur_x:cur_x + self.patch_size, cur_y:cur_y + self.patch_size])
        else:
            cur_mask = np.copy(self.open_set_mask[cur_x:cur_x + self.patch_size, cur_y:cur_y + self.patch_size])

        # shifting mask patch on the fly - old, deprecated
        # cur_mask, openset_mask = self.shift_patch_mask(cur_mask)

        # sanity check
        assert len(cur_path[0]) == self.patch_size and len(cur_path[0][0]) == self.patch_size, \
            "Wrong patch size " + str(len(cur_path[0])) + " x " + str(len(cur_path[0][0]))

        # normalization and data augmentation
        cur_path = (cur_path / 255) - 0.5
        if self.mode == 'train':
            cur_path, cur_mask, _ = data_augmentation(cur_path, cur_mask)
        cur_path = np.transpose(cur_path, (0, 3, 1, 2))

        # Turning to tensors.
        cur_path = torch.from_numpy(cur_path.copy())
        cur_mask = torch.from_numpy(cur_mask.copy())

        return cur_path.float(), cur_mask.long(), np.array([cur_x, cur_y])

    def __len__(self):
        return len(self.distr_data)

