import datetime
import os
import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class LossHistory():
    def __init__(self, log_dir, model, input_shape):

        # Create a timestamped log directory
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        self.log_dir = os.path.join(log_dir, "metrics_" + str(time_str))
        self.losses = []
        self.val_loss = []
        self.train_accuracies = []
        self.val_accuracies = []

        # Create the log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

        # Try adding the model graph to TensorBoard
        try:
            dummy_input = torch.randn(2, 2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except Exception as e:
            print(f"Error adding graph to TensorBoard: {e}")

    def append_metrics(self, epoch, loss, val_loss, train_accuracy, val_accuracy):

        # Ensure the log directory exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Append metrics to the lists
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        self.train_accuracies.append(train_accuracy)
        self.val_accuracies.append(val_accuracy)

        # Save metrics to a text file
        with open(os.path.join(self.log_dir, "epoch_metrics.txt"), 'a') as f:
            f.write(f"{loss},{val_loss},{train_accuracy},{val_accuracy}\n")

        # Log metrics to TensorBoard
        self.writer.add_scalar('train_loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.writer.add_scalar('train_accuracy', train_accuracy, epoch)
        self.writer.add_scalar('val_accuracy', val_accuracy, epoch)

        # Generate and save plots
        self.metrics_plot()

    def metrics_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        plt.plot(iters, self.train_accuracies, 'blue', linewidth=2, label='train accuracy')
        plt.plot(iters, self.val_accuracies, 'purple', linewidth=2, label='val accuracy')

        try:
            # Apply smoothing if there are at least 5 points
            if len(self.losses) >= 5:  # Ensure a minimum window length of 5
                num = 5 if len(self.losses) < 25 else 15

                plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                         label='smooth train loss')
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--',
                         linewidth=2, label='smooth val loss')
        except Exception as e:
            print(f"Error smoothing plot: {e}")

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.legend(loc="upper right")

        # Save the plot to the log directory
        plt.savefig(os.path.join(self.log_dir, "epoch_metrics.png"))

        plt.cla()
        plt.close("all")

