import torch, os
import numpy as np
from omniglotNShot import OmniglotNShot
import argparse

from meta import Meta
from tensorboardX import SummaryWriter

# Main function to execute the training and validation process
def main(args):

    # Setting random seeds for reproducibility
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    # Print the provided arguments
    print(args)

    # Define the configuration of the neural network
    config = [
        ('conv2d', [64, 1, 3, 3, 2, 0]), # Convolutional layer
        ('relu', [True]),                # ReLU activation function
        ('bn', [64]),                    # Batch normalization
        ('conv2d', [64, 64, 3, 3, 2, 0]),# Convolutional layer
        ('relu', [True]),                # ReLU activation function
        ('bn', [64]),                    # Batch normalization
        ('conv2d', [64, 64, 3, 3, 2, 0]),# Convolutional layer
        ('relu', [True]),                # ReLU activation function
        ('bn', [64]),                    # Batch normalization
        ('conv2d', [64, 64, 2, 2, 1, 0]),# Convolutional layer
        ('relu', [True]),                # ReLU activation function
        ('bn', [64]),                    # Batch normalization
        ('flatten', []),                 # Flatten layer
        ('linear', [args.n_way, 64])     # Fully connected layer
    ]

    # Set the device to GPU
    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    # Count the total number of trainable parameters
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # Create an instance of the dataset for training
    db_train = OmniglotNShot('omniglot',
                       batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       imgsz=args.imgsz)

    # Create a SummaryWriter for TensorBoard logging
    tb = SummaryWriter('runs')

    # Training loop
    for step in range(args.epoch):

        # Get the next batch of support and query sets
        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

        # Perform a training step
        loss, accs = maml(x_spt, y_spt, x_qry, y_qry)

        # Log training metrics to TensorBoard every 100 steps
        if step % 100 == 0:
            tb.add_scalar('Train/Accuracy', accs.mean(), step)
            tb.add_scalar('Train/Loss', loss.mean(), step)
            print('step:', step, '\ttraining acc:', accs, '\ttraining loss:', loss.mean())

        # Perform validation every 100 steps
        if step % 100 == 0:
            val_losses, val_accs = [], []
            for _ in range(1000 // args.task_num):
                # Get the next batch of support and query sets for validation
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                # Fine-tune and evaluate the model on the validation set
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_loss, test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    val_losses.append(test_loss.cpu())
                    val_accs.append(test_acc)

            # Log validation metrics to TensorBoard
            val_losses = np.array([loss.item() for loss in val_losses]).mean()
            val_accs = np.array(val_accs).mean()
            tb.add_scalar('Validation/Accuracy', val_accs, step)
            tb.add_scalar('Validation/Loss', val_losses, step)
            print('Validation acc:', val_accs, '\tValidation loss:', val_losses)

    # Close the TensorBoard writer
    tb.close()

# Argument parser to take command line arguments
if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=5000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=1)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.05)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    # Parse arguments and call the main function
    args = argparser.parse_args()
    main(args)
