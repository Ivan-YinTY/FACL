import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from eval.evaluation import evaluate_verification
from torchvision.utils import save_image
import utils
import numpy as np


class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.m = params['m']
        self.lr_scheduler = params["lr_scheduler"]
        self.args = params['args']
        # self.batch_size = params['batch_size']
        # self.num_workers = params['num_workers']
        # self.checkpoint_interval = params['checkpoint_interval']

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, trainloader, testloader):
        loss_stats = utils.AverageMeter()
        self.initializes_target_network()
        best_result, best_snapshot = None, None
        for epoch in range(self.max_epochs):
            for batch_idx, ((batch_view_1, batch_view_2), _) in enumerate(trainloader):
                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)
                sample = torch.cat((batch_view_1, batch_view_2), -1)
                save_image(sample[:10], "results/%d.bmp" % (batch_idx % 64), nrow=1, normalize=True, range=(-1.0, 1.0))
                loss = self.update(batch_view_1, batch_view_2)

                # simclr
                # f_view_1 = self.online_network(batch_view_1)[1]
                # f_view_2 = self.online_network(batch_view_2)[1]
                # f_view_1_nml = F.normalize(f_view_1, p=2, dim=1)
                # f_view_2_nml = F.normalize(f_view_2, p=2, dim=1)
                # p = f_view_1_nml.matmul(f_view_2_nml.T)
                # s = 20
                # loss_1 = -1 * F.log_softmax(s * p, dim=1).diag().mean()
                # loss_2 = -1 * F.log_softmax(s * p, dim=0).diag().mean()
                # loss = (loss_1 + loss_2) / 2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network_parameters()  # update the key encoder

                loss_stats.update(loss.item())
                print(utils.dt(), 'Epoch:[%d]-[%d/%d] batchLoss:%.4f averLoss:%.4f' %
                      (epoch, batch_idx, len(trainloader), loss_stats.val, loss_stats.avg))

            roc, aver, auc = evaluate_verification(self.online_network, testloader)
            self.lr_scheduler.step()
            # save the current best model based on eer
            best_result, best_snapshot = \
                self.save_model(self.online_network, {'metrics': roc, 'aver': roc[0], 'epoch': epoch}, best_result, best_snapshot)
            print("End of epoch {}".format(epoch))

        print(utils.dt(), 'Training completed.')
        print(utils.dt(), '------------------Best Results---------------------')
        epoch, roc = best_result['epoch'], best_result['metrics']
        print(utils.dt(),
              'EER: %.2f%%, FPR100:%.2f%%, FPR1000:%.2f%%, FPR10000:%.2f%%, FPR0:%.2f%%, Aver: %.2f%% @ epoch %d' %
              (roc[0] * 100, roc[1] * 100, roc[2] * 100, roc[3] * 100, roc[4] * 100, np.mean(roc) * 100, epoch))
        # save checkpoints
        # self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1)[1])
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2)[1])

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)[1]
            targets_to_view_1 = self.target_network(batch_view_2)[1]

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1.detach())
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2.detach())
        return loss.mean()

    # def save_model(self, PATH):
    #
    #     torch.save({
    #         'online_network_state_dict': self.online_network.state_dict(),
    #         'target_network_state_dict': self.target_network.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #     }, PATH)


    def save_model(self, model, current_result, best_result, best_snapshot, lower_is_better=True):
        aver = current_result['aver']
        epoch = current_result['epoch']
        args = self.args
        prefix = 'seed=%d_dataset=%s_network=%s_loss=%s' % (args.seed, args.dataset, args.network, args.loss)
        # save the current best model
        if best_result is None or (aver >= best_result['aver'] and not lower_is_better) \
                or (aver <= best_result['aver'] and lower_is_better):
            best_result = current_result
            snapshot = {'model': model.module.state_dict() if torch.cuda.device_count() > 1 and args.multi_gpu else model.state_dict(),
                        'epoch': epoch,
                        'args': args
                        }
            if best_snapshot is not None:
                os.system('rm %s' % (best_snapshot))

            best_snapshot = './snapshots/%s_Best%s=%.2f_Epoch=%d.pth' % (
                prefix, 'ROC' if lower_is_better else 'CMC', aver * 100, epoch)
            torch.save(snapshot, best_snapshot)
        # always save the final model
        if epoch == args.max_epoch - 1:
            snapshot = {'model': model.module.state_dict() if torch.cuda.device_count() > 1 and args.multi_gpu else model.state_dict(),
                        'epoch': epoch,
                        'args': args
                        }
            last_snapshot = './snapshots/%s_Final%s=%.2f_Epoch=%d.pth' % (
                prefix, 'ROC' if lower_is_better else 'CMC', aver * 100, epoch)
            torch.save(snapshot, last_snapshot)
        return best_result, best_snapshot