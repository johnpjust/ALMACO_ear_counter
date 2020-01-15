import numpy as np
from windowed_frames import windowed
import torch

def batch(iterable, device, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield torch.from_numpy(iterable[ndx:min(ndx + n, l)]).float().to(device)

def inference(model, data_loader_train, data_loader_val, data_loader_test, args, empty_logs_train, empty_logs_val, empty_logs_test):

    train_pred_counts = []
    train_empty_counts = []
    train_static_counts = []
    train_y_counts = []
    model.eval()
    for ind in range(len(data_loader_train)):

        for i_ in range(3): ## normally range(2) when using empty logs during training
            if i_ == 0:
                x_mb = windowed(data_loader_train[ind][0], n=args.num_frames, step=1)
                y_mb = data_loader_train[ind][1]
            elif i_ == 1:
                x_mb = windowed(data_loader_train[ind][0][empty_logs_train[ind] == 0], n=args.num_frames, step=1)
                y_mb = 0
            else:
                x_mb = np.repeat(data_loader_train[ind][0][:,:,:,:,None], args.num_frames, axis=-1)
                x_mb = np.moveaxis(x_mb, -1,1)
                y_mb = 0

            count = []
            model.zero_grad()
            torch.cuda.empty_cache()
            with torch.no_grad():
                for x in batch(x_mb, args.device, args.batch_size):
                    x = x.permute(0, 1, -1, -3, -2)
                    count.extend(np.array(model(x).cpu().numpy()))

            if i_ == 0:
                train_pred_counts.append(np.array(count))
                train_y_counts.append(y_mb)
            elif i_ == 1:
                train_empty_counts.append(np.array(count))
            else:
                train_static_counts.append(np.array(count))

    validation_pred = []
    validation_pred_static = []
    validation_pred_empty = []
    validation_y_counts = []
    for ind in range(len(data_loader_val)):
        for i_ in range(3):
            if i_ == 0:
                x_mb = windowed(data_loader_val[ind][0], n=args.num_frames, step=1)
                y_mb = data_loader_val[ind][1]
            elif i_ == 1:
                x_mb = windowed(data_loader_val[ind][0][empty_logs_val[ind] == 0], n=args.num_frames, step=1)
            else:
                x_mb = np.repeat(data_loader_val[ind][0][:, :, :, :, None], args.num_frames, axis=-1)
                x_mb = np.moveaxis(x_mb, -1, 1)
            count = []
            model.zero_grad()
            torch.cuda.empty_cache()
            with torch.no_grad():
                for x in batch(x_mb, args.device, args.batch_size):
                    x = x.permute(0, 1, -1, -3, -2)
                    count.extend(np.array(model(x).cpu().numpy()))

                if i_ == 0:
                    validation_pred.append(np.array(count))
                    validation_y_counts.append(y_mb)
                elif i_ == 1:
                    validation_pred_empty.append(np.array(count))
                else:
                    validation_pred_static.append(np.array(count))

    test_pred = []
    test_pred_static = []
    test_pred_empty = []
    test_y_counts = []
    for ind in range(len(data_loader_test)):
        for i_ in range(3):
            if i_ == 0:
                x_mb = windowed(data_loader_test[ind][0], n=args.num_frames, step=1)
                y_mb = data_loader_test[ind][1]
            elif i_ == 1:
                x_mb = windowed(data_loader_test[ind][0][empty_logs_test[ind] == 0], n=args.num_frames, step=1)
            else:
                x_mb = np.repeat(data_loader_test[ind][0][:, :, :, :, None], args.num_frames, axis=-1)
                x_mb = np.moveaxis(x_mb, -1, 1)

            count = []
            model.zero_grad()
            torch.cuda.empty_cache()
            with torch.no_grad():
                for x in batch(x_mb, args.device, args.batch_size):
                    x = x.permute(0, 1, -1, -3, -2)
                    count.extend(np.array(model(x).cpu().numpy()))

                if i_ == 0:
                    test_pred.append(np.array(count))
                    test_y_counts.append(y_mb)
                elif i_ == 1:
                    test_pred_empty.append(np.array(count))
                else:
                    test_pred_static.append(np.array(count))

    return [[train_pred_counts, validation_pred, test_pred], [train_y_counts, validation_y_counts, test_y_counts],
            [train_empty_counts, validation_pred_empty, test_pred_empty], [train_static_counts, validation_pred_static, test_pred_static]]