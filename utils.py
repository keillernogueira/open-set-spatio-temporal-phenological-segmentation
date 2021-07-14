import os
import argparse

import torch


class TwoWayDict(dict):
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2


cedro = TwoWayDict({0: (0, 255, 255),
                    1: (0, 0, 255),
                    2: (255, 0, 0),
                    3: (0, 255, 0),
                    4: (0, 0, 0)})


def str2bool(v):
    """
    Function to transform strings into booleans.

    v: string variable
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_best_models(net, optimizer, output_path, best_records, epoch, acc, acc_cls, cm):
    if len(best_records) < 5:
        best_records.append({'epoch': epoch, 'acc': acc, 'acc_cls': acc_cls, 'cm': cm})

        torch.save(net.state_dict(), os.path.join(output_path, 'model_' + str(epoch) + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(output_path, 'opt_' + str(epoch) + '.pth'))
    else:
        # find min saved acc
        min_index = 0
        for i, r in enumerate(best_records):
            if best_records[min_index]['acc_cls'] > best_records[i]['acc_cls']:
                min_index = i
        # check if currect acc is greater than min saved acc
        if acc_cls > best_records[min_index]['acc_cls']:
            # if it is, delete previous files
            min_step = str(best_records[min_index]['epoch'])

            os.remove(os.path.join(output_path, 'model_' + min_step + '.pth'))
            os.remove(os.path.join(output_path, 'opt_' + min_step + '.pth'))

            # replace min value with current
            best_records[min_index] = {'epoch': epoch, 'acc': acc, 'acc_cls': acc_cls, 'cm': cm}

            # save current model
            torch.save(net.state_dict(), os.path.join(output_path, 'model_' + str(epoch) + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(output_path, 'opt_' + str(epoch) + '.pth'))
