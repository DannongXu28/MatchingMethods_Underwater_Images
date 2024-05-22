import os

import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import get_lr


def fit_one_epoch(model_train, model, loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                  Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss = 0
    total_accuracy = 0

    val_loss = 0
    val_total_accuracy = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    # Set model to training mode
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

        # Clear optimizer gradient
        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(images)
            loss_value = loss(outputs, targets)

            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(images)
                loss_value = loss(outputs, targets)
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            # Calculate accuracy
            preds = torch.round(nn.Sigmoid()(outputs))
            accuracy = torch.mean(torch.eq(preds, targets).float())

        total_loss += loss_value.item()
        total_accuracy += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'train_accuracy': total_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    # Set model to evaluation mode
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

            optimizer.zero_grad()
            outputs = model_train(images)
            loss_value = loss(outputs, targets)

            # Calculate accuracy
            preds = torch.round(nn.Sigmoid()(outputs))
            accuracy = torch.mean(torch.eq(preds, targets).float())

        val_loss += loss_value.item()
        val_total_accuracy += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                'val_accuracy': val_total_accuracy / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        # Log the metrics to TensorBoard
        loss_history.append_metrics(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val,
                                    total_accuracy / epoch_step, val_total_accuracy / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        # Save model weights periodically or at the last epoch
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
            epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        # Save the best model based on validation loss
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights_test1.pth"))

        # Save the last model weights
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights_test1.pth"))
