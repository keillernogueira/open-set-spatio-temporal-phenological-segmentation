import sys
import datetime
import joblib

from networks import GRSL, FCN8s
from dataloader import DataLoader, PatchDataLoader, load_images, create_distributions, create_patch_distributions
from utils import *
from openpcs import *

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def test(test_loader, net, epoch, ignore_index=8):
    # Setting network for evaluation mode.
    net.eval()

    all_labels = None
    all_preds = None
    all_pos = None
    all_softs = None
    with torch.no_grad():
        # Iterating over batches.
        for i, data in enumerate(test_loader):
            # Obtaining images, labels and paths for batch.
            inps, labs, pos = data[0], data[1], data[2]
            inps_c = Variable(inps).cuda()

            # Forwarding.
            outs, _, _ = net(inps_c.permute(1, 0, 2, 3, 4))
            # Computing probabilities.
            soft_outs = F.softmax(outs, dim=1)
            # Obtaining prior predictions.
            prds = soft_outs.cpu().data.numpy().argmax(axis=1)

            if all_labels is None:
                all_softs = soft_outs.cpu().data.numpy()
                all_labels = labs
                all_preds = prds
                all_pos = pos
            else:
                all_softs = np.concatenate((all_softs, soft_outs.cpu().data.numpy()))
                all_labels = np.concatenate((all_labels, labs))
                all_preds = np.concatenate((all_preds, prds))
                all_pos = np.concatenate((all_pos, pos))

        if type(net) is not GRSL:  # not pixelwise
            valid_labels = all_labels[all_labels != ignore_index].flatten()
            valid_preds = all_preds[all_labels != ignore_index].flatten()
            acc = accuracy_score(valid_labels, valid_preds)
            bacc = balanced_accuracy_score(valid_labels, valid_preds)
            conf_m = confusion_matrix(valid_labels, valid_preds)
        else:
            acc = accuracy_score(all_labels, all_preds)
            bacc = balanced_accuracy_score(all_labels, all_preds)
            conf_m = confusion_matrix(all_labels, all_preds)

        print(" ---- Validation/Test -- Epoch " + str(epoch) +
              " -- Time " + str(datetime.datetime.now().time()) +
              " Overall Accuracy= " + "{:.4f}".format(acc) +
              " Normalized Accuracy= " + "{:.4f}".format(bacc) +
              " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
              )
        sys.stdout.flush()

    return acc, bacc, conf_m, all_labels, all_preds, all_pos, all_softs


def train(train_loader, net, criterion, optimizer, epoch, ignore_index=8):
    net.train()
    train_loss = list()
    for i, data in enumerate(train_loader):
        # Obtaining data and labels
        inputs, labels = data[0], data[1]

        # Casting tensors to cuda.
        inputs_c, labels_c = inputs.cuda(), labels.cuda()
        inps = Variable(inputs_c).cuda()
        labs = Variable(labels_c).cuda()

        # Clears the gradients of optimizer.
        optimizer.zero_grad()

        # Forwarding.
        outs, _, _ = net(inps.permute(1, 0, 2, 3, 4))  # permute to change the branches with the batch size
        soft_outs = F.softmax(outs, dim=1)

        # Obtaining predictions.
        prds = soft_outs.cpu().data.numpy().argmax(axis=1)
        # print(inps.permute(1, 0, 2, 3, 4).shape, labs.shape, prds.shape)

        # Computing loss.
        loss = criterion(outs, labs)

        # Computing backpropagation.
        loss.backward()
        optimizer.step()

        # Updating loss meter.
        train_loss.append(loss.data.item())

        # Printing.
        if i > 0 and i % 5 == 0:
            if type(net) is not GRSL:  # not pixelwise
                labels = labels.numpy().flatten()
                prds = prds.flatten()
                valid_labels = labels[labels != ignore_index]
                valid_preds = prds[labels != ignore_index]
                acc = accuracy_score(valid_labels, valid_preds)
                bacc = balanced_accuracy_score(valid_labels, valid_preds)
                conf_m = confusion_matrix(valid_labels, valid_preds)
            else:
                acc = accuracy_score(labels, prds)
                bacc = balanced_accuracy_score(labels, prds)
                conf_m = confusion_matrix(labels, prds)

            print("Training -- Epoch " + str(epoch) + " -- Iter " + str(i+1) + "/" + str(len(train_loader)) +
                  " -- Time " + str(datetime.datetime.now().time()) +
                  " -- Training Minibatch: Loss= " + "{:.6f}".format(train_loss[-1]) +
                  " Overall Accuracy= " + "{:.4f}".format(acc) +
                  " Normalized Accuracy= " + "{:.4f}".format(bacc) +
                  " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
                  )

    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')

    # general options
    parser.add_argument('--operation', type=str, required=True,
                        help='Operation', choices=['train', 'test', 'openset'])
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save outcomes (such as images and trained models) of the algorithm.')

    # dataset options
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset path.')
    parser.add_argument('--images', type=str, nargs="+", required=True, help='Image/timestamp names.')
    parser.add_argument('--patch_size', type=int, required=False, default=25, help='Patch size.')
    parser.add_argument('--hidden_class', type=int, required=True,
                        help='Hidden class for open-set. Values from 0 to 3.')

    # model options
    parser.add_argument('--network', type=str, required=True,
                        help='Network model. grsl option is deprecated.', choices=['grsl', 'fcn'])
    parser.add_argument('--model_path', type=str, required=False, default=None,
                        help='Path to a trained model to be used during the inference.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epoch_num', type=int, default=50, help='Number of epochs')
    args = parser.parse_args()
    print(args)
    assert args.network == 'grsl' or args.network == 'fcn'

    images, mask = load_images(args.dataset_path, args.images, args.patch_size if args.network == 'grsl' else None)
    print(images.shape, mask.shape)

    if os.path.isfile(os.path.join(os.path.abspath(os.getcwd()), args.network + '_train_data.npy')):
        train_data = np.load(args.network + '_train_data.npy', allow_pickle=True)
        test_data = np.load(args.network + '_test_data.npy', allow_pickle=True)
        openset_data = np.load(args.network + '_openset_data.npy', allow_pickle=True)
    else:
        if args.network == 'grsl':
            train_data, test_data, openset_data = create_distributions(mask, 4)
        else:
            train_data, test_data, openset_data = create_patch_distributions(mask, args.patch_size,
                                                                             args.hidden_class, num_classes=4)
        np.save(args.network + '_train_data.npy', train_data)
        np.save(args.network + '_test_data.npy', test_data)
        np.save(args.network + '_openset_data.npy', openset_data)
    print(train_data.shape, test_data.shape, openset_data.shape)

    # data loaders
    if args.network == 'grsl':
        train_dataset = DataLoader('train', images, mask, train_data, args.patch_size)
        test_dataset = DataLoader('test', images, mask, test_data, args.patch_size)
    else:
        train_dataset = PatchDataLoader('train', images, mask, train_data, args.patch_size, args.hidden_class)
        test_dataset = PatchDataLoader('test', images, mask, test_data, args.patch_size, args.hidden_class)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=4, drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=4, drop_last=False)

    # network
    if args.network == 'grsl':
        model = GRSL(len(args.images), 3, train_dataset.num_classes).cuda()
        criterion = nn.CrossEntropyLoss().cuda()  # ignore class 8
    elif args.network == 'fcn':
        model = FCN8s(len(args.images), 3, train_dataset.num_classes - 1).cuda()
        criterion = nn.CrossEntropyLoss(ignore_index=8).cuda()  # ignore class 8
    else:
        raise NotImplementedError("Network " + args.network + " not implemented")

    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                           betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    if args.operation == 'train':
        curr_epoch = 1
        best_records = []
        print("Training")
        for epoch in range(curr_epoch, args.epoch_num + 1):
            train(train_dataloader, model, criterion, optimizer, epoch)

            # Computing test.
            acc, bacc, cm, _, _, _, _ = test(test_dataloader, model, epoch)
            save_best_models(model, optimizer, args.output_path, best_records, epoch, acc, bacc, cm, num_saves=1)

            scheduler.step()
    elif args.operation == 'test':
        assert args.model_path is not None, "For inference, flag model_path should be set."
        ckpt = torch.load(args.model_path)
        model.load_state_dict(ckpt)
        model.cuda()

        epoch_num = int(os.path.splitext(os.path.basename(args.model_path))[0].split('_')[-1])
        _, _, _, all_labels, all_preds, all_pos, all_softs = test(test_dataloader, model, epoch_num)
        # print(np.asarray(all_labels).shape, np.asarray(all_preds).shape,
        #       np.asarray(all_pos).shape, np.asarray(all_softs).shape)
        all_labels[all_labels <= 4] = 1
        all_labels[all_labels > 4] = 0

        _, h, w, c = test_dataset.images.shape
        prob_image = np.zeros((4, h, w), dtype=np.float32)
        occur_im = np.zeros((4, h, w), dtype=np.float32)
        for i, p in enumerate(all_pos):
            prob_image[1:4, p[0]:p[0]+args.patch_size, p[1]:p[1]+args.patch_size] += all_softs[i, :, :, :] * all_labels[i]
            occur_im[:, p[0]:p[0]+args.patch_size, p[1]:p[1]+args.patch_size] += 1
        occur_im[np.where(occur_im == 0)] = 1
        pred_image = np.argmax(prob_image / occur_im.astype(float), axis=0)

        # print(np.bincount(pred_image.flatten()))
        pred_image[pred_image == 0] = 4
        for i in range(1, args.hidden_class+1):
            pred_image[pred_image == i] = i - 1
        # print(np.bincount(pred_image.flatten()))

        color_image = lookup_class[pred_image]
        # Saving predictions.
        imageio.imwrite(os.path.join(args.output_path, 'prediction_closed_set_epoch_' + str(epoch_num) + '.png'),
                        color_image)
        evaluate_map(mask, pred_image, args.hidden_class)
    elif args.operation == 'openset':
        assert args.model_path is not None, "For OpenPCS, flag model_path should be set."

        model.load_state_dict(torch.load(args.model_path))
        model.cuda()
        if os.path.isfile(os.path.join(args.output_path, args.network + '_model_pca.pkl')):
            print('loading trained openset model')
            model_full = joblib.load(os.path.join(args.output_path, args.network + '_model_pca.pkl'))
        else:
            print('training openset')
            model_full = train_openset(train_dataloader, model)
            joblib.dump(model_full, os.path.join(args.output_path, args.network + '_model_pca.pkl'))

        print('thresholds', model_full['thresholds'])
        # test
        open_dataset = PatchDataLoader('open', images, mask, np.concatenate((test_data, openset_data)),
                                       args.patch_size, args.hidden_class)
        open_dataloader = torch.utils.data.DataLoader(open_dataset, batch_size=args.batch_size,
                                                      shuffle=False, num_workers=4, drop_last=False)
        test_openset(open_dataloader, model, model_full, args.output_path)
    else:
        raise NotImplementedError("Operation " + args.operation + " not implemented")
