import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.triplet_selector import RandomNegativeTripletSelector


class CosFace(nn.Module):
    def __init__(self, s=20.0, m=0.2, verbal=False):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m
        self.iter = 0
        self.verbal = verbal

    def forward(self, input, labels):
        # input: size = B x num_class
        cos = input
        one_hot = torch.zeros(cos.size()).cuda()
        # one_hot = torch.zeros(cos.size(), device='cuda')
        one_hot = one_hot.scatter_(1, torch.tensor(labels.view(-1, 1), dtype=torch.int64), 1)
        output = self.s * (cos - one_hot * self.m)

        softmax_output = F.log_softmax(output, dim=1)
        loss = -1 * softmax_output.gather(1, torch.tensor(labels.view(-1, 1),dtype=torch.int64))
        loss = loss.mean()

        if self.iter % 10 == 0 and self.verbal:
            angles = cos.data.acos()
            angles_non_target = torch.sum((1 - one_hot) * angles, dim=1) / (angles.shape[1] - 1)
            angles_non_target_mean = angles_non_target.mean()
            angles_non_target_min = angles_non_target.min()
            angles_non_target_max = angles_non_target.max()

            angles_target = angles.gather(1, torch.tensor(labels.view(-1, 1),dtype=torch.int64))
            angles_target_mean = angles_target.mean()
            angles_target_min = angles_target.min()
            angles_target_max = angles_target.max()
            print("angle_target:%f (min:%f, max:%f), angle_non_target:%f (min:%f, max:%f)" %
                  (angles_target_mean, angles_target_min, angles_target_max, angles_non_target_mean, angles_non_target_min, angles_non_target_max))
        self.iter += 1
        return loss


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin=0.2, s=20.0, is_distance=True):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = RandomNegativeTripletSelector(margin=margin, is_distance=is_distance)
        self.is_distance = is_distance
        self.s = s
        self.iter = 0

    def forward(self, embeddings, target):
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        triplets = self.triplet_selector.get_triplets(embeddings_normalized, target)

        if embeddings_normalized.is_cuda:
            triplets = triplets.cuda()
        if self.is_distance:
            ap_distances = (embeddings_normalized[triplets[:, 0]] - embeddings_normalized[triplets[:, 1]]).pow(2).sum(1)
            an_distances = (embeddings_normalized[triplets[:, 0]] - embeddings_normalized[triplets[:, 2]]).pow(2).sum(1)
            # pn_distances = (embeddings_normalized[triplets[:, 1]] - embeddings_normalized[triplets[:, 2]]).pow(2).sum(1)
            # losses = F.relu(ap_distances - torch.min(an_distances, pn_distances) + self.margin)
            losses = F.relu(ap_distances - an_distances + self.margin)
            # angle_intra = torch.mean(torch.acos(-0.5*(ap_distances.detach() - 2)))
            # angle_inter = torch.mean(torch.acos(-0.5*(an_distances.detach() - 2)))
            # if self.iter % 10 == 0:
            #     print('angle_intra: %.2f, angle_inter: %.2f' % (angle_intra, angle_inter))
        else:
            ap_distances = (embeddings_normalized[triplets[:, 0]] * embeddings_normalized[triplets[:, 1]]).sum(1)
            an_distances = (embeddings_normalized[triplets[:, 0]] * embeddings_normalized[triplets[:, 2]]).sum(1)
            # pn_distances = (embeddings_normalized[triplets[:, 1]] * embeddings_normalized[triplets[:, 2]]).sum(1)
            # losses = torch.log(1 + torch.exp(self.t * (torch.max(an_distances, pn_distances) - ap_distances)))
            losses = torch.log(1 + torch.exp(self.s * (an_distances - ap_distances + self.margin)))
            # losses = 2*F.relu(an_distances - ap_distances + self.margin)
        self.iter += 1
        return losses.mean(), len(triplets)
