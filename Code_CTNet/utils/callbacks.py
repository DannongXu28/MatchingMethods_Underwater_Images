import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# Class to log and visualize training/validation metrics
class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = os.path.join(log_dir, "metrics_" + str(time_str))
        self.losses     = []
        self.val_loss   = []
        self.train_accuracies = []
        self.val_accuracies = []

        # Create directory for log files
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)

        # Add model graph to TensorBoard
        try:
            dummy_input     = torch.randn(2, 2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    # Append metrics for the current epoch and log them
    def append_metrics(self, epoch, loss, val_loss, train_accuracy, val_accuracy):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)
        self.train_accuracies.append(train_accuracy)
        self.val_accuracies.append(val_accuracy)

        # Log to TensorBoard
        self.writer.add_scalar('Loss/Train', loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        self.writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        self.writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

        # Save metrics to text files
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_train_accuracy.txt"), 'a') as f:
            f.write(str(train_accuracy))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_accuracy.txt"), 'a') as f:
            f.write(str(val_accuracy))
            f.write("\n")

        # Plot and save the metrics
        self.plot_metrics()

    # Plot training and validation metrics
    def plot_metrics(self):
        iters = range(len(self.losses))

        # Plot Loss
        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='Train Loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='Validation Loss')
        try:
            num = 5 if len(self.losses) < 25 else 15
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2, label='Smoothed Train Loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2, label='Smoothed Validation Loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")

        # Plot Accuracy
        plt.figure()
        plt.plot(iters, self.train_accuracies, 'blue', linewidth=2, label='Train Accuracy')
        plt.plot(iters, self.val_accuracies, 'orange', linewidth=2, label='Validation Accuracy')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.log_dir, "epoch_accuracy.png"))
        plt.cla()
        plt.close("all")
