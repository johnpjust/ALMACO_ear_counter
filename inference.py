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
            for x in batch(x_mb, args.device, args.batch_size):
                x = x.permute(0, 1, -1, -3, -2)
                count.extend(model(x))

            if i_ == 0:
                train_pred_counts.append(count)
            elif i_ == 1:

    model.eval()
    # train_loss = tf.reduce_mean(train_loss)
    validation_loss = []
    validation_loss_static = []
    validation_loss_empty = []
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
            count = 0
            model.zero_grad()
            torch.cuda.empty_cache()
            with torch.no_grad():
                for x in batch(x_mb, args.device, args.batch_size):
                    x = x.permute(0, 1, -1, -3, -2)
                    count += model(x).sum()

                if i_ == 0:
                    validation_loss.append((count.cpu().numpy() - y_mb)**2)
                elif i_ == 1:
                    validation_loss_empty.append(count.cpu().numpy()**2)
                else:
                    validation_loss_static.append(count.cpu().numpy()**2)

    validation_loss = np.mean(validation_loss)
    validation_loss_empty = np.mean(validation_loss_empty)
    validation_loss_static = np.mean(validation_loss_static)
    # print("validation loss:  " + str(validation_loss))

    test_loss=[]
    test_loss_static = []
    test_loss_empty = []
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

            count = 0
            model.zero_grad()
            torch.cuda.empty_cache()
            with torch.no_grad():
                for x in batch(x_mb, args.device, args.batch_size):
                    x = x.permute(0, 1, -1, -3, -2)
                    count += model(x).sum()

                if i_ == 0:
                    test_loss.append((count.cpu().numpy() - y_mb)**2)
                elif i_ == 1:
                    test_loss_empty.append(count.cpu().numpy()**2)
                else:
                    test_loss_static.append(count.cpu().numpy()**2)

    test_loss = np.mean(test_loss)
    test_loss_empty = np.mean(test_loss_empty)
    test_loss_static = np.mean(test_loss_static)
    # print("test loss:  " + str(test_loss))