# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""SCAN model"""

from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import torch.nn.init
import torchvision.models as models
from apex import amp
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm

from modeling_bertnewsinglecut import BertModelNew


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def l1norm(X, dim, eps=1e-5):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-5):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).add(eps).sqrt() + eps
    X = torch.div(X, norm)
    return X


class EncoderCross(nn.Module):
    def __init__(self):
        super(EncoderCross, self).__init__()
        dropout = 0.1
        self.opt = opt
        self.margin = 0.2

    def forward(self, scores, n_img, n_cap, test=False):
        scores = scores.view(n_img, n_cap)
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        # keep the maximum violating negative for each query
        eps = 1e-5
        cost_s = cost_s.pow(4).sum(1).add(eps).sqrt().sqrt()  # .sqrt()#.div(cost_s.size(1)).mul(2)
        cost_im = cost_im.pow(4).sum(0).add(eps).sqrt().sqrt()  # .sqrt()#.div(cost_im.size(0)).mul(2)
        return cost_s.sum() + cost_im.sum()


class EncoderText(nn.Module):

    def __init__(self, opt):
        super(EncoderText, self).__init__()
        self.encoder = BertModelNew.from_pretrained('bert/')
        self.encoder2 = BertModelNew.from_pretrained('bert/')

        self.fc = nn.Linear(2048, 768)
        self.fc2 = nn.Linear(768, 30600)
        self.fc3 = nn.Linear(768, 1601)
        self.norm = nn.LayerNorm(768, eps=1e-5)
        self.relu = nn.ReLU()
        self.mlm = opt.mlm
        self.cm = opt.cm
        self.mrm = opt.mrm

        # self.ff = EncoderLayerMinus(768,768,0.1)

    def calPrec(self, pred, grnd):
        idx = grnd != -1
        pred = pred[idx]
        grnd = grnd[idx]
        corr = pred == grnd
        # prec = corr.sum()/corr.size(0)
        return corr.sum() * 1.0, corr.size(0) * 1.0  # prec

    def sample_negative(self, vision_feat, text_output, non_pad_mask, vision_mask, head_mask, offset):
        offset = min(offset, vision_feat.size(0))
        index = [i for i in range(offset, vision_feat.size(0))] + [i for i in range(offset)]
        # index = [i for i in range(1,vision_feat.size(0))] + [0]
        vision_feat = vision_feat[index]
        vision_mask = vision_mask[index]
        catfeat = torch.cat([text_output, vision_feat], 1)
        vision_mask_cat = torch.cat([non_pad_mask, vision_mask], 1).squeeze()
        extended_attention_mask_cat = vision_mask_cat.squeeze()[:, None, None, :]
        extended_attention_mask_cat = (1.0 - extended_attention_mask_cat) * -10000.0
        catnewn = self.encoder2.encoder(catfeat, extended_attention_mask_cat, head_mask)
        catnewn = catnewn[0]
        text_out = catnewn[:, 0]  # .view(bs,bs,-1)
        img_out = catnewn[:, text_output.size(1):].sum(1)
        return text_out, img_out

    def forward(self, input_ids, token_type_ids, non_pad_mask, vision_feat, vision_mask, gt_labels=None,
                vision_labels=None, MLM=False, istest=False):
        text_output = self.encoder.embeddings(input_ids=input_ids, position_ids=None,
                                              token_type_ids=token_type_ids.long().squeeze())
        head_mask = [None] * 20
        vision_feat = self.fc(vision_feat)
        vision_feat = self.norm(vision_feat)

        bs = text_output.size(0)
        tl = text_output.size(1)
        vl = vision_feat.size(1)

        if istest == False and MLM == False:
            text_output = text_output.unsqueeze(0).expand(bs, -1, -1, -1).contiguous().view(bs * bs, tl, -1)
            vision_feat = vision_feat.unsqueeze(1).expand(-1, bs, -1, -1).contiguous().view(bs * bs, vl, -1)
            non_pad_mask = non_pad_mask.unsqueeze(0).expand(bs, -1, -1).contiguous().view(bs * bs, -1)
            vision_mask = vision_mask.unsqueeze(1).expand(-1, bs, -1).contiguous().view(bs * bs, -1)

        catfeat = torch.cat([text_output, vision_feat], 1)
        vision_mask_cat = torch.cat([non_pad_mask, vision_mask], 1).squeeze()
        extended_attention_mask_cat = vision_mask_cat.squeeze()[:, None, None, :]
        extended_attention_mask_cat = (1.0 - extended_attention_mask_cat) * -10000.0

        catnew = self.encoder2.encoder(catfeat, extended_attention_mask_cat, head_mask)
        catnew = catnew[0]
        if MLM:
            if self.cm:
                text_global_neg1, img_global_neg1 = self.sample_negative(vision_feat, text_output, non_pad_mask,
                                                                         vision_mask, head_mask, 1)
                text_global_neg2, img_global_neg2 = self.sample_negative(vision_feat, text_output, non_pad_mask,
                                                                         vision_mask, head_mask, 2)
                text_global_neg3, img_global_neg3 = self.sample_negative(vision_feat, text_output, non_pad_mask,
                                                                         vision_mask, head_mask, 3)
            # text_global_neg4, img_global_neg4 = self.sample_negative(vision_feat,text_output,non_pad_mask,vision_mask,head_mask,4)
            # text_global_neg5, img_global_neg5 = self.sample_negative(vision_feat,text_output,non_pad_mask,vision_mask,head_mask,5)

            text_out = catnew[:, :text_output.size(1)]
            img_out = catnew[:, text_output.size(1):]
            text_global_pos = catnew[:, 0]  # .view(bs,bs,-1)
            img_global_pos = img_out.sum(1)

            scores_pos = cosine_similarity(text_global_pos, img_global_pos, -1).view(-1, 1)
            if self.cm:
                scores_neg1 = cosine_similarity(text_global_neg1, img_global_neg1, -1).view(-1, 1)
                scores_neg2 = cosine_similarity(text_global_neg2, img_global_neg2, -1).view(-1, 1)
                scores_neg3 = cosine_similarity(text_global_neg3, img_global_neg3, -1).view(-1, 1)
                # scores_neg4 =  cosine_similarity(text_global_neg4,img_global_neg4,-1).view(-1,1)
                # scores_neg5 =  cosine_similarity(text_global_neg5,img_global_neg5,-1).view(-1,1)
                margin = 0.2
                scores = torch.cat([scores_pos, scores_neg1, scores_neg2, scores_neg3], 1)
                scores_pos = scores[:, 0]  # .view(-1,1)
                d1 = scores_pos.expand_as(scores)
                cost_s = (margin + scores - d1).clamp(min=0)
                cost_s[:, 0] = 0
                eps = 1e-5
                cost_s = cost_s.max(1)[0]  # cost_s.pow(8).sum(1).add(eps).sqrt().sqrt().sqrt()
                # scores = F.softmax(scores*5,dim=1)
                # scores_pos = scores[:,0]
                # loss3 = scores_pos.log().mul(-1).sum().div(catnew.size(0))
                loss3 = cost_s.sum().div(cost_s.size(0))
                # scores_all = torch.cat([scores_pos,scores_neg1,scores_neg2,scores_neg3],1)
                # scores_all = F.softmax(scores_all.mul(5),dim=1)
                # scores_pos = scores_all[:,0]
                # loss3 = scores_pos.log().mul(-1).sum().div(catnew.size(0))

            pre_labels = self.fc2(text_out)
            pre_labels_vis = self.fc3(img_out)
            pre_vis = torch.argmax(pre_labels_vis, dim=-1).view(-1)
            pre_txt = torch.argmax(pre_labels, dim=-1).view(-1)

            corr, total = self.calPrec(pre_txt, gt_labels.view(-1))

            corr_vis, total_vis = self.calPrec(pre_vis, vision_labels.view(-1))

            loss1 = F.cross_entropy(pre_labels.view((-1, pre_labels.size()[-1])),
                                    gt_labels.view(-1),
                                    ignore_index=-1)

            loss2 = F.cross_entropy(pre_labels_vis.view((-1, pre_labels_vis.size()[-1])),
                                    vision_labels.view(-1),
                                    ignore_index=-1)

            corr = loss1 / loss1 * corr
            total = loss1 / loss1 * total

            corr_vis = loss1 / loss1 * corr_vis
            total_vis = loss1 / loss1 * total_vis

            # print(corr/total)
            loss = 0
            if self.mlm:
                loss += loss1
            if self.mrm:
                loss += loss2
            if self.cm:
                loss += loss3
            # loss loss1 + loss2 #+ loss3#2 + loss3
            return loss, corr, total, corr_vis, total_vis  # + loss2

        text_out = catnew[:, 0]  # .view(bs,bs,-1)
        vision_output = catnew[:, text_output.size(1):].sum(1)  # .view(bs,bs,-1)
        scores = cosine_similarity(vision_output, text_out, -1)
        margin = 0.2
        if istest:
            return scores  # .view(bs,bs,-1),vision_output.view(bs,bs,-1)#[0]
        else:
            scores = scores.view(bs, bs)
            diagonal = scores.diag().view(scores.size(0), 1)
            d1 = diagonal.expand_as(scores)
            d2 = diagonal.t().expand_as(scores)

            cost_s = (margin + scores - d1).clamp(min=0)
            cost_im = (margin + scores - d2).clamp(min=0)
            mask = torch.eye(scores.size(0)) > .5
            I = Variable(mask).cuda()
            cost_s = cost_s.masked_fill_(I, 0)
            cost_im = cost_im.masked_fill_(I, 0)


            eps = 1e-5
            # loss for img retrieve text
            cost_s = cost_s.pow(4).sum(1).add(eps).sqrt().sqrt()  # .sqrt()#.div(cost_s.size(1)).mul(2)
            # loss for text retrieve image
            cost_im = cost_im.pow(4).sum(0).add(eps).sqrt().sqrt()  # .sqrt()#.div(cost_im.size(0)).mul(2)
            return cost_s.sum() + cost_im.sum()


def cosine_similarity(x1, x2, dim=1, eps=1e-5):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def xattn_score_t2i(images, captions, cap_lens, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)

    return similarities


def xattn_score_i2t(images, captions, cap_lens, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities


class SCAN(object):
    """
    Stacked Cross Attention Network (SCAN) model
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip

        self.txt_enc = EncoderText(opt)

        self.drop = torch.nn.Dropout(p=0.0)

        if torch.cuda.is_available():
            self.txt_enc.cuda()  # .cuda()
            cudnn.benchmark = True

        params = list(self.txt_enc.parameters())
        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.txt_enc, self.optimizer = amp.initialize(self.txt_enc, self.optimizer, opt_level="O1")
        self.txt_enc = torch.nn.DataParallel(self.txt_enc)
        # Loss and Optimizer
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.txt_enc.load_state_dict(state_dict[0])

    def train_start(self):
        """switch to train mode
        """
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.txt_enc.eval()

    def forward_emb(self, images, captions, target_mask, vision_mask, volatile=False, istest=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images.float(), volatile=volatile)
        captions = torch.LongTensor(captions)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()  # .cuda()
            captions = captions.cuda()  # .cuda()
        # Forward

        n_img = images.size(0)
        n_cap = captions.size(0)
        if istest:
            images = images.unsqueeze(1).expand(n_img, n_cap, images.size(1),
                                                images.size(2)).contiguous().view(-1, images.size(1), images.size(2))
            captions = captions.unsqueeze(0).expand(n_img, n_cap, captions.size(1)).contiguous().view(-1,
                                                                                                      captions.size(1))

        attention_mask = get_non_pad_mask(captions).cuda().squeeze()
        token_type_ids = torch.zeros_like(attention_mask)

        video_non_pad_mask = get_non_pad_mask(vision_mask).cuda().squeeze()
        if istest:
            video_non_pad_mask = video_non_pad_mask.unsqueeze(1).expand(n_img, n_cap, images.size(1)).contiguous().view(
                -1, images.size(1))

        scores = self.txt_enc(captions, token_type_ids, attention_mask, images, video_non_pad_mask, istest)
        return scores

    def forward_embMLM(self, images, captions, target_mask, vision_mask, gt_labels, vision_labels, volatile=False,
                       istest=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images.float(), volatile=volatile)
        captions = torch.LongTensor(captions)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()  # .cuda()
            captions = captions.cuda()  # .cuda()
        # Forward
        n_img = images.size(0)
        n_cap = captions.size(0)

        attention_mask = get_non_pad_mask(captions).cuda().squeeze()
        token_type_ids = torch.zeros_like(attention_mask)

        video_non_pad_mask = get_non_pad_mask(vision_mask).cuda().squeeze()
        loss, corr, total, corr_vis, total_vis = self.txt_enc(captions, token_type_ids, attention_mask, images,
                                                              video_non_pad_mask, gt_labels, vision_labels, MLM=True)

        self.logger.update('MLM', corr.sum() / total.sum(), n_cap)
        self.logger.update('MRM', corr_vis.sum() / total_vis.sum(), n_img)

        return loss

    def forward_loss(self, img_emb, cap_emb, cap_len, text_non_pad_mask, text_slf_attn_mask, img_non_pad_mask,
                     img_slf_attn_mask, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        scores = self.cross_att(img_emb, cap_emb, cap_len, text_non_pad_mask, text_slf_attn_mask, img_non_pad_mask,
                                img_slf_attn_mask)
        loss = self.criterion(scores)
        self.logger.update('Le', loss.item(), scores.size(0))
        return loss

    def train_emb(self, images, captions, target_mask, vision_mask, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # measure accuracy and record loss
        scores = self.forward_emb(images, captions, target_mask, vision_mask)
        # measure accuracy and record loss

        self.optimizer.zero_grad()
        if scores is not None:
            loss = scores.sum()
        else:
            return
        # compute gradient and do SGD step
        # loss.backward()
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

    def train_embMLM(self, images, captions, target_mask, vision_mask, gt_labels, vision_labels, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        gt_labels = gt_labels.long().cuda()
        # measure accuracy and record loss
        loss = self.forward_embMLM(images, captions, target_mask, vision_mask, gt_labels, vision_labels)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = loss.sum().div(loss.size(0))
        # compute gradient and do SGD step
        # loss.backward()
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
