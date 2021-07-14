import datetime

from networks import GRSL
from dataloader import DataLoader
from config import *
from utils import *
from openpcs import *

from sklearn.metrics import accuracy_score, confusion_matrix

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def test(test_loader, net, epoch):
    # Setting network for evaluation mode.
    net.eval()

    all_labels = None
    all_preds = None
    all_pos = None
    with torch.no_grad():
        # Iterating over batches.
        for i, data in enumerate(test_loader):

            # Obtaining images, labels and paths for batch.
            inps, labs, pos = data[0], data[1], data[2]

            inps = inps.squeeze()
            labs = labs.squeeze()

            # Casting to cuda variables.
            inps_c = Variable(inps).cuda()
            # labs_c = Variable(labs).cuda()

            # Forwarding.
            outs, _, _ = net(inps_c.permute(1, 0, 2, 3, 4))
            # Computing probabilities.
            soft_outs = F.softmax(outs, dim=1)

            # Obtaining prior predictions.
            prds = soft_outs.cpu().data.numpy().argmax(axis=1)

            if all_labels is None:
                all_labels = labs
                all_preds = prds
                all_pos = pos
            else:
                all_labels = np.concatenate((all_labels, labs))
                all_preds = np.concatenate((all_preds, prds))
                all_pos = np.concatenate((all_pos, pos))

        acc = accuracy_score(all_labels, all_preds)
        conf_m = confusion_matrix(all_labels, all_preds)

        _sum = 0.0
        for k in range(len(conf_m)):
            _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)

        print("Validation/Test -- Epoch " + str(epoch) +
              " -- Time " + str(datetime.datetime.now().time()) +
              " Overall Accuracy= " + "{:.4f}".format(acc) +
              " Normalized Accuracy= " + "{:.4f}".format(_sum / float(outs.shape[1])) +
              " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
              )

        sys.stdout.flush()

    return acc, _sum / float(outs.shape[1]), conf_m, all_labels, all_preds, all_pos


def train(train_loader, net, criterion, optimizer, epoch):
    # Setting network for training mode.
    net.train()

    # Average Meter for batch loss.
    train_loss = list()

    # Iterating over batches.
    for i, data in enumerate(train_loader):
        # Obtaining data and labels
        inputs, labels = data[0], data[1]
        # print(inputs.shape, labels)

        # Casting tensors to cuda.
        inputs_c, labels_c = inputs.cuda(), labels.cuda()
        inputs_c.squeeze_(0)
        labels_c.squeeze_(0)

        # Casting to cuda variables.
        inps = Variable(inputs_c).cuda()
        labs = Variable(labels_c).cuda()

        # Clears the gradients of optimizer.
        optimizer.zero_grad()

        # Forwarding.
        outs, _, _ = net(inps.permute(1, 0, 2, 3, 4))  # permute to change the branches with the batch size
        soft_outs = F.softmax(outs, dim=1)

        # Obtaining predictions.
        prds = soft_outs.cpu().data.numpy().argmax(axis=1)
        # print(soft_outs.data, prds, type(prds))
        # print(inps.permute(1, 0, 2, 3, 4).shape, labs.shape, prds.shape)

        # Computing loss.
        loss = criterion(outs, labs)

        # Computing backpropagation.
        loss.backward()
        optimizer.step()

        # Updating loss meter.
        train_loss.append(loss.data.item())

        # Printing.
        if (i + 1) % DISPLAY_STEP == 0:
            acc = accuracy_score(labels, prds)
            conf_m = confusion_matrix(labels, prds)

            _sum = 0.0
            for k in range(len(conf_m)):
                _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)

            print("Training -- Epoch " + str(epoch) + " -- Iter " + str(i+1) + "/" + str(len(train_loader)) +
                  " -- Time " + str(datetime.datetime.now().time()) +
                  " -- Training Minibatch: Loss= " + "{:.6f}".format(train_loss[-1]) +
                  " Overall Accuracy= " + "{:.4f}".format(acc) +
                  " Normalized Accuracy= " + "{:.4f}".format(_sum / float(outs.shape[1])) +
                  " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
                  )

    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')

    # general options
    parser.add_argument('--operation', type=str, required=True,
                        help='Operation. Options: [Train | Test | OpenPCS]')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save outcomes (such as images and trained models) of the algorithm.')

    # dataset options
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset path.')
    parser.add_argument('--images', type=str, nargs="+", required=True, help='Image/timestamp names.')
    parser.add_argument('--patch_size', type=int, required=False, default=25, help='Patch size.')

    # model options
    parser.add_argument('--network', type=str, required=True,
                        help='Network model. Options: [grsl].')
    parser.add_argument('--model_path', type=str, required=False, default=None,
                        help='Path to a trained model to be used during the inference.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epoch_num', type=int, default=50, help='Number of epochs')
    args = parser.parse_args()
    print(args)

    images, mask = DataLoader.load_images(args.dataset_path, args.images, args.patch_size)
    print(images.shape, mask.shape)

    if os.path.isfile(os.path.join(os.path.abspath(os.getcwd()), 'train_data.npy')):
        train_data = np.load(os.path.join(os.path.abspath(os.getcwd()), 'train_data.npy'), allow_pickle=True)
        test_data = np.load(os.path.join(os.path.abspath(os.getcwd()), 'test_data.npy'), allow_pickle=True)
        no_data = np.load(os.path.join(os.path.abspath(os.getcwd()), 'no_data.npy'), allow_pickle=True)
    else:
        train_data, test_data, no_data = DataLoader.create_distributions_over_pixel_classes(mask, 4)
        print(train_data.shape, test_data.shape, no_data.shape)
        np.save(os.path.join(os.path.abspath(os.getcwd()), 'train_data.npy'), train_data)
        np.save(os.path.join(os.path.abspath(os.getcwd()), 'test_data.npy'), test_data)
        np.save(os.path.join(os.path.abspath(os.getcwd()), 'no_data.npy'), no_data)

    # data loaders
    train_dataset = DataLoader('Train', images, mask, train_data, args.patch_size)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=NUM_WORKERS, drop_last=False)

    test_dataset = DataLoader('Test', images, mask, test_data, args.patch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

    # network
    if args.network == 'grsl':
        model = GRSL(len(args.images), 3, train_dataset.num_classes).cuda()
        optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                               betas=(0.9, 0.99))
    else:
        raise NotImplementedError("Network " + args.network + " not implemented")

    # loss
    criterion = nn.CrossEntropyLoss().cuda()

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    if args.operation == 'Train':
        curr_epoch = 1
        best_records = []
        print("Training")
        for epoch in range(curr_epoch, args.epoch_num + 1):
            train(train_dataloader, model, criterion, optimizer, epoch)
            if epoch % VAL_INTERVAL == 0:
                # Computing test.
                acc, nacc, cm, _, _, _ = test(test_dataloader, model, epoch)

                save_best_models(model, optimizer, args.output_path, best_records, epoch, acc, nacc, cm)

            scheduler.step()
    elif args.operation == 'Test':
        assert args.model_path is not None, "For inference, flag model_path should be set."
        ckpt = torch.load(args.model_path)
        model.load_state_dict(ckpt)
        model.cuda()

        epoch_num = int(os.path.splitext(os.path.basename(args.model_path))[0].split('_')[-1])
        _, _, _, _, all_preds, all_pos = test(test_dataloader, model, epoch_num)

        _, h, w, c = test_dataset.images.shape
        pred_image = np.zeros((h, w, c), dtype=np.uint8)
        for i, p in enumerate(all_pos):
            pred_image[p[0], p[1], :] = cedro[all_preds[i]]

        # Saving predictions.
        imageio.imwrite(os.path.join(args.output_path, 'prediction_closed_set_epoch' + str(epoch_num) + '.png'),
                        pred_image)

    elif args.operation == 'OpenPCS':
        assert args.model_path is not None, "For OpenPCS, flag model_path should be set."

        all_dataset = DataLoader('OpenPCS', images, mask,
                                 np.concatenate((train_data, test_data, no_data)), args.patch_size)
        all_dataloader = torch.utils.data.DataLoader(all_dataset, batch_size=args.batch_size,
                                                     shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        ckpt = torch.load(args.model_path)
        model.load_state_dict(ckpt)
        model.cuda()
        if os.path.isfile(os.path.join(args.output_path, 'model_pca.pkl')):
            model_full = joblib.load(os.path.join(args.output_path, 'model_pca.pkl'))
        else:
            model_full = train_openset(train_dataloader, model, train_dataset.num_classes, args.output_path)
        test_openset(all_dataloader, model, model_full, all_dataset, args.output_path)
    else:
        raise NotImplementedError("Operation " + args.operation + " not implemented")
