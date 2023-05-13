import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.epsilon = args.epsilon

    def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels):
        """
        Cross-Modal Projection Matching Loss(CMPM)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
            i2t_loss: cmpm loss for image projected to text
            t2i_loss: cmpm loss for text projected to image
            pos_avg_sim: average cosine-similarity for positive pairs
            neg_avg_sim: averate cosine-similarity for negative pairs
        """

        batch_size = image_embeddings.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)

        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        image_proj_text = torch.matmul(image_embeddings, text_norm.t())
        text_proj_image = torch.matmul(text_embeddings, image_norm.t())

        # normalize the true matching distribution
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

        i2t_pred = F.softmax(image_proj_text, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + self.epsilon)) # (4)
        t2i_pred = F.softmax(text_proj_image, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + self.epsilon))

        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        return cmpm_loss

    def compute_triplet_loss(self, image_pos, text, image_neg):
        distance_pos = F.pairwise_distance(image_pos, text, p=2)
        distance_neg = F.pairwise_distance(image_neg, text, p=2)

        losses = F.relu(distance_pos - distance_neg + 5)
        return losses.mean()

    def forward(self, img_f3_1, img_f4_1, img_f41_1, img_f42_1, img_f43_1, img_f44_1, img_f45_1, img_f46_1,
            img_f3_2, img_f4_2, img_f41_2, img_f42_2, img_f43_2, img_f44_2, img_f45_2, img_f46_2,
            txt_f3, txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46
            ):
        loss = 0.0

        loss = self.compute_triplet_loss(img_f3_1, txt_f3, img_f3_2) \
                + self.compute_triplet_loss(img_f41_1, txt_f41, img_f41_2) \
                + self.compute_triplet_loss(img_f42_1, txt_f42, img_f42_2) \
                + self.compute_triplet_loss(img_f43_1, txt_f43, img_f43_2) \
                + self.compute_triplet_loss(img_f44_1, txt_f44, img_f44_2) \
                + self.compute_triplet_loss(img_f45_1, txt_f45, img_f45_2) \
                + self.compute_triplet_loss(img_f46_1, txt_f46, img_f46_2) \
                + self.compute_triplet_loss(img_f4_1, txt_f4, img_f4_2)

        return loss
