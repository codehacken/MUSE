# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Important notes:
https://github.com/facebookresearch/MUSE/issues/135
"""

import os
from logging import getLogger
import time
import numpy as np
import scipy
import scipy.linalg
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from .utils import get_optimizer, load_embeddings, normalize_embeddings, export_embeddings
from .utils import clip_parameters
from .dico_builder import build_dictionary
from .evaluation.word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary, COM_DIC_EVAL_PATH
from .rcsls_loss import RCSLS

logger = getLogger()


class Trainer(object):

    def __init__(self, src_emb, tgt_emb, mapping, discriminator, params, rev_discriminator=None):
        """
        Initialize trainer script.
        """
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_dico = params.src_dico
        self.tgt_dico = getattr(params, 'tgt_dico', None)
        self.mapping = mapping
        self.discriminator = discriminator
        self.rev_discriminator = rev_discriminator
        self.params = params

        # RCSLS implementation.
        if self.params.loss == "r" or self.params.loss == "mr":
            self.criterionRCSLS = RCSLS()

        # optimizers.
        if hasattr(params, 'map_optimizer'):
            optim_fn, optim_params = get_optimizer(params.map_optimizer)
            self.map_optimizer = optim_fn(mapping.parameters(), **optim_params)
        if hasattr(params, 'dis_optimizer'):
            optim_fn, optim_params = get_optimizer(params.dis_optimizer)
            self.dis_optimizer = optim_fn(discriminator.parameters(), **optim_params)

            if params.bidirectional and rev_discriminator is not None:
                optim_fn, optim_params = get_optimizer(params.dis_optimizer)
                self.rev_dis_optimizer = optim_fn(rev_discriminator.parameters(), **optim_params)

        else:
            assert discriminator is None

        # best validation score
        self.best_valid_metric = -1e12

        self.decrease_lr = False

    def get_dis_xy(self, volatile, reverse=False):
        """
        Get discriminator input batch / output target.
        """
        # select random word IDs
        bs = self.params.batch_size
        mf = self.params.dis_most_frequent
        assert mf <= min(len(self.src_dico), len(self.tgt_dico))
        src_ids = torch.LongTensor(bs).random_(len(self.src_dico) if mf == 0 else mf)
        tgt_ids = torch.LongTensor(bs).random_(len(self.tgt_dico) if mf == 0 else mf)
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        # get word embeddings
        if reverse:
            # Reverse target and source.
            with torch.no_grad():
                src_emb = self.tgt_emb(Variable(tgt_ids))
                tgt_emb = self.src_emb(Variable(src_ids))
                tgt_emb = Variable(tgt_emb.data)
            if volatile:
                with torch.no_grad():
                    src_emb = self.mapping(Variable(src_emb.data), fdir=False)
            else:
                src_emb = self.mapping(Variable(src_emb.data), fdir=False)
        else:
            with torch.no_grad():
                src_emb = self.src_emb(Variable(src_ids))
                tgt_emb = self.tgt_emb(Variable(tgt_ids))
                tgt_emb = Variable(tgt_emb.data)
            if volatile:
                with torch.no_grad():
                    src_emb = self.mapping(Variable(src_emb.data))
            else:
                src_emb = self.mapping(Variable(src_emb.data))

        # input / target
        if reverse:
            x = torch.cat([tgt_emb, src_emb], 0)
        else:
            x = torch.cat([src_emb, tgt_emb], 0)

        y = torch.FloatTensor(2 * bs).zero_()
        y[:bs] = 1 - self.params.dis_smooth
        y[bs:] = self.params.dis_smooth
        y = Variable(y.cuda() if self.params.cuda else y)

        return x, y

    def dis_step(self, stats, reverse=False):
        """
        Train the discriminator.
        """
        self.discriminator.train()

        # loss
        x, y = self.get_dis_xy(volatile=True)
        preds = self.discriminator(Variable(x.data))
        loss = F.binary_cross_entropy(preds, y)
        stats['DIS_COSTS'].append(loss.data.item())

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (discriminator)")
            exit()

        # optim
        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()
        clip_parameters(self.discriminator, self.params.dis_clip_weights)

    def dis_r_step(self, stats):
        """
        Train the discriminator.
        """
        self.rev_discriminator.train()

        # loss
        x, y = self.get_dis_xy(volatile=True, reverse=True)
        preds = self.rev_discriminator(Variable(x.data))
        loss = F.binary_cross_entropy(preds, y)
        stats['DIS_COSTS'].append(loss.data.item())

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (Reverse discriminator)")
            exit()

        # optim
        self.rev_dis_optimizer.zero_grad()
        loss.backward()
        self.rev_dis_optimizer.step()
        clip_parameters(self.rev_discriminator, self.params.dis_clip_weights)

    def mapping_step(self, stats, parallel=False, reverse=False):
        """
        Fooling discriminator training step.
        """
        if self.params.dis_lambda == 0:
            return 0

        if reverse:
            self.rev_discriminator.eval()
        else:
            self.discriminator.eval()

        # loss.
        x, y = self.get_dis_xy(volatile=False, reverse=reverse)

        if reverse:
            preds = self.rev_discriminator(x)
        else:
            preds = self.discriminator(x)

        loss = F.binary_cross_entropy(preds, 1 - y)
        loss = self.params.dis_lambda * loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (fool discriminator)")
            exit()

        # optim
        self.map_optimizer.zero_grad()
        loss.backward()
        self.map_optimizer.step()
        # self.orthogonalize(parallel)

        return 2 * self.params.batch_size

    def load_training_dico(self, dico_train, size=0, descending=True):
        """
        Load training dictionary.
        """
        word2id1 = self.src_dico.word2id
        word2id2 = self.tgt_dico.word2id

        # identical character strings
        if dico_train == "identical_char":
            self.dico = load_identical_char_dico(word2id1, word2id2)
        # use one of the provided dictionary
        elif dico_train == "default":
            filename = '%s-%s.0-5000.txt' % (self.params.src_lang, self.params.tgt_lang)
            self.dico = load_dictionary(
                os.path.join(DIC_EVAL_PATH, filename),
                word2id1, word2id2, size=size,
                descending=descending
            )
        elif dico_train == "combined":
            filename = '%s-%s.0-5000.txt' % (self.params.src_lang, self.params.tgt_lang)
            self.dico = load_dictionary(
                os.path.join(COM_DIC_EVAL_PATH, filename),
                word2id1, word2id2, size=size,
                descending=descending
            )
        # dictionary provided by the user
        else:
            self.dico = load_dictionary(dico_train, word2id1, word2id2)

        # cuda
        if self.params.cuda:
            self.dico = self.dico.cuda()

    def build_dictionary(self):
        """
        Build a dictionary from aligned embeddings.
        """
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        self.dico = build_dictionary(src_emb, tgt_emb, self.params)

    def procrustes(self):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        A = self.src_emb.weight.data[self.dico[:, 0]]
        B = self.tgt_emb.weight.data[self.dico[:, 1]]
        W = self.mapping.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))

    def bdma_procrustes(self, with_reverse=False):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        A = self.src_emb.weight.data[self.dico[:, 0]]
        B = self.tgt_emb.weight.data[self.dico[:, 1]]
        W = self.mapping.module.layers[0].weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))

        if with_reverse:
            logger.info("Performing reverse procrustes....")
            W = self.mapping.module.reverse[0].weight.data
            M = A.transpose(0, 1).mm(B).cpu().numpy()
            U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
            W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))


    def bdma_step(self, stats):
        """
        Train the mapping.
        """
        self.mapping.train()

        # Create batches.
        A = self.src_emb.weight.data[self.dico[:, 0]]
        B = self.tgt_emb.weight.data[self.dico[:, 1]]

        if self.params.loss == "r" or self.params.loss == "mr":
            neg_src_emb = self.src_emb.weight.data
            neg_tgt_emb = self.tgt_emb.weight.data

        # Shuffle.
        r = torch.randperm(A.shape[0])
        A = A[r]
        B = B[r]

        bs = self.params.batch_size
        num_batches = int(A.shape[0] / bs)

        if num_batches == 0:
            num_batches = 1

        if self.params.loss == "r" or self.params.loss == "mr":
            logger.info("Using RCSLS Loss...")

        avg_f_loss = 0.0; avg_b_loss = 0.0;
        for i in range(0, num_batches):
            s = i * bs
            e = (i + 1) * bs
            x, y = A[s : e], B[s : e]

            # Forward optimization.
            # Zero Grad.
            self.map_optimizer.zero_grad()

            # Predictions.
            f_preds = self.mapping(x)
            if self.params.loss == "m" or self.params.loss == "mr":
                f_loss = F.mse_loss(f_preds, y)
            else:
                f_loss = 0

            if self.params.loss == "r" or self.params.loss == "mr":
                # Negative Sampling.
                neg_src_emb_trans = self.mapping(neg_src_emb)
                f_rcsls_loss = self.criterionRCSLS(x, f_preds, y, neg_src_emb,
                                                   neg_src_emb_trans, neg_tgt_emb)
                f_loss += f_rcsls_loss

            # optimizer.
            f_loss.backward()
            self.map_optimizer.step()
            clip_parameters(self.mapping, self.params.map_clip_weights)

            # Reverse optimization.
            if self.params.bidirectional:
                self.map_optimizer.zero_grad()
                b_preds = self.mapping(y, fdir=False)

                if self.params.loss == "m" or self.params.loss == "mr":
                    b_loss = self.params.n_rev_beta * F.mse_loss(b_preds, x)
                else:
                    b_loss = 0

                if self.params.loss == "r" or self.params.loss == "mr":
                    # Negative Sampling.
                    neg_tgt_emb_trans = self.mapping(neg_tgt_emb, fdir=False)
                    b_rcsls_loss = self.criterionRCSLS(y, b_preds, x,
                                                       neg_tgt_emb,
                                                       neg_tgt_emb_trans,
                                                       neg_src_emb)
                    b_loss += self.params.n_rev_beta * b_rcsls_loss

                b_loss.backward()
                self.map_optimizer.step()
                clip_parameters(self.mapping, self.params.map_clip_weights)
                avg_b_loss += b_loss.cpu()

            # Collect loss information.
            stats['MSE_COSTS'].append(f_loss.data.item())
            avg_f_loss += f_loss.cpu()

        self.mapping.eval()
        self.orthogonalize_bdma()
        return avg_f_loss / num_batches, avg_b_loss / num_batches, num_batches


    def bdma_unsup_step(self, stats):
        """
        Train the mapping.
        """
        # self.mapping.train()

        # # Create batches.
        # A = self.src_emb.weight.data
        # B = self.tgt_emb.weight.data
        #
        # # if self.params.loss == "r" or self.params.loss == "mr":
        # #     neg_src_emb = self.src_emb.weight.data
        # #     neg_tgt_emb = self.tgt_emb.weight.data
        #
        # # Shuffle.
        # # NOTE: Source and Target are NOT aligned.
        # # r = torch.randperm(A.shape[0])
        # # A = A[r]
        # # B = B[r]
        #
        # bs = self.params.batch_size
        # num_batches = int(A.shape[0] / bs)
        #
        # # if num_batches == 0:
        # #     num_batches = 1
        # #
        # # if self.params.loss == "r" or self.params.loss == "mr":
        # #     logger.info("Using RCSLS Loss...")
        #
        # avg_f_loss = 0.0; avg_b_loss = 0.0;
        # print("Number Of Batches: {}".format(num_batches))
        # for i in range(0, num_batches):
        #     s = i * bs
        #     e = (i + 1) * bs
        #     x, y = A[s : e], B[s : e]
        #
        #     # Forward optimization.
        #     # Predictions.
        #     f_preds = self.mapping(self.mapping(x), fdir=False)
        #     if self.params.loss == "m" or self.params.loss == "mr":
        #         f_loss = F.mse_loss(f_preds, x)
        #     else:
        #         f_loss = 0
        #
        #     # if self.params.loss == "r" or self.params.loss == "mr":
        #     #     # Negative Sampling.
        #     #     neg_src_emb_trans = self.mapping(neg_src_emb)
        #     #     # f_rcsls_loss = self.criterionRCSLS(x, f_preds, y, neg_src_emb, neg_src_emb_trans, neg_tgt_emb)
        #     #     f_rcsls_loss = self.criterionRCSLS(x, f_preds, x, neg_src_emb, neg_src_emb_trans, neg_src_emb)
        #     #     f_loss += f_rcsls_loss
        #
        #     # optimizer.
        #     # Zero Grad.
        #     self.map_optimizer.zero_grad()
        #     f_loss.backward()
        #     self.map_optimizer.step()
        #     clip_parameters(self.mapping, self.params.map_clip_weights)
        #
        #     # Collect loss information.
        #     stats['MSE_COSTS'].append(f_loss.data.item())
        #     avg_f_loss += f_loss.cpu()





        #     # Reverse optimization.
        #     if self.params.bidirectional:
        #         b_preds = self.mapping(self.mapping(y, fdir=False))
        #
        #         if self.params.loss == "m" or self.params.loss == "mr":
        #             b_loss = self.params.n_rev_beta * F.mse_loss(b_preds, y)
        #         else:
        #             b_loss = 0
        #
        #         # if self.params.loss == "r" or self.params.loss == "mr":
        #         #     # Negative Sampling.
        #         #     neg_tgt_emb_trans = self.mapping(neg_tgt_emb, fdir=False)
        #         #     b_rcsls_loss = self.criterionRCSLS(y, b_preds, x, neg_tgt_emb, neg_tgt_emb_trans, neg_src_emb)
        #         #     b_loss += self.params.n_rev_beta * b_rcsls_loss
        #
        #         self.map_optimizer.zero_grad()
        #         b_loss.backward()
        #         self.map_optimizer.step()
        #         clip_parameters(self.mapping, self.params.map_clip_weights)
        #         avg_b_loss += b_loss.cpu()
        #
        # self.mapping.eval()
        # # self.orthogonalize_bdma()

        """
        Fooling discriminator training step.
        """
        n_words_proc = 0
        tic = time.time()
        for n_iter in range(0, self.params.epoch_size, self.params.batch_size):
            # Discriminator training.
            # Forward Training.
            for _ in range(self.params.dis_steps):
                self.dis_step(stats)

            # if self.params.bidirectional:
            #     for _ in range(self.params.dis_steps):
            #         self.dis_r_step(stats)

            # Mapping Step.
            n_words_proc += self.mapping_step(stats)

            # Reverse Training.
            # if self.params.bidirectional:
            #     n_words_proc += self.mapping_step(stats, reverse=True)

            if n_iter % 500 == 0:
                stats_str = [('DIS_COSTS', 'Discriminator loss')]
                stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                             for k, v in stats_str if len(stats[k]) > 0]
                stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
                logger.info(('%06i - ' % n_iter) + ' - '.join(stats_log))

                # reset
                tic = time.time()
                n_words_proc = 0
                for k, _ in stats_str:
                    del stats[k][:]

        # return avg_f_loss / num_batches, \
        #        avg_b_loss / num_batches, num_batches
        return 0.0, 0.0, 0.0


    def orthogonalize(self, parallel=False):
        """
        Orthogonalize the mapping.
        """
        if self.params.map_beta > 0:
            if parallel:
                W = self.mapping.module.layers[0].weight.data
            else:
                W = self.mapping.weight.data

            beta = self.params.map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

    def orthogonalize_bdma(self):
        """
        Orthogonalize the BDMA mapping.
        """

        if self.params.map_beta > 0:
            beta = self.params.map_beta
            for weight in self.mapping.module.all_weights:
                W = weight.data
                W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

    def update_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.map_optimizer:
            return
        old_lr = self.map_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing learning rate: %.8f -> %.8f" % (old_lr, new_lr))
            self.map_optimizer.param_groups[0]['lr'] = new_lr

        if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:
            if to_log[metric] < self.best_valid_metric:
                logger.info("Validation metric is smaller than the best: %.5f vs %.5f"
                            % (to_log[metric], self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    old_lr = self.map_optimizer.param_groups[0]['lr']
                    self.map_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                    logger.info("Shrinking the learning rate: %.5f -> %.5f"
                                % (old_lr, self.map_optimizer.param_groups[0]['lr']))
                self.decrease_lr = True

    def save_best(self, to_log, metric):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if to_log[metric] > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = to_log[metric]
            logger.info('* Best value for "%s": %.5f' % (metric, to_log[metric]))
            # save the mapping
            W = self.mapping.weight.data.cpu().numpy()
            path = os.path.join(self.params.exp_path, 'best_mapping.pth')
            logger.info('* Saving the mapping to %s ...' % path)
            torch.save(W, path)

    def reload_best(self):
        """
        Reload the best mapping.
        """
        path = os.path.join(self.params.exp_path, 'best_mapping.pth')
        logger.info('* Reloading the best model from %s ...' % path)
        # reload the model
        assert os.path.isfile(path)
        to_reload = torch.from_numpy(torch.load(path))
        W = self.mapping.weight.data
        assert to_reload.size() == W.size()
        W.copy_(to_reload.type_as(W))


    def save_best_bdma(self, to_log, metric):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if to_log[metric] > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = to_log[metric]
            logger.info('* Best value for "%s": %.5f' % (metric, to_log[metric]))

            # save the mapping.
            path = os.path.join(self.params.exp_path, 'best_mapping.pth')
            logger.info('* Saving the mapping to %s ...' % path)
            torch.save(self.mapping.state_dict(), path)

    def reload_best_bdma(self):
        """
        Reload the best mapping.
        """
        path = os.path.join(self.params.exp_path, 'best_mapping.pth')
        logger.info('* Reloading the best model from %s ...' % path)
        # reload the model
        assert os.path.isfile(path)
        self.mapping.load_state_dict(torch.load(path))

    def export(self, bdma_model=True):
        """
        Export embeddings.
        """
        params = self.params

        # load all embeddings
        logger.info("Reloading all embeddings for mapping ...")
        params.src_dico, src_emb = load_embeddings(params, source=True, full_vocab=True)
        params.tgt_dico, tgt_emb = load_embeddings(params, source=False, full_vocab=True)

        # apply same normalization as during training
        normalize_embeddings(src_emb, params.normalize_embeddings, mean=params.src_mean)
        normalize_embeddings(tgt_emb, params.normalize_embeddings, mean=params.tgt_mean)

        # Create copies of src_emb and tgt_emb
        fin_src_emb = torch.from_numpy(np.zeros(src_emb.shape))
        fin_tgt_emb = torch.from_numpy(np.zeros(tgt_emb.shape))

        # map source embeddings to the target space
        bs = 512
        logger.info("Map source embeddings to the target space ...")
        for i, k in enumerate(range(0, len(src_emb), bs)):
            x = Variable(src_emb[k:k + bs], volatile=True)
            fin_src_emb[k:k + bs] = self.mapping(x.cuda() if params.cuda else x).data.cpu()

        # write embeddings to the disk (SRC to TGT)
        export_embeddings(fin_src_emb, tgt_emb, params)

        # map target embeddings to the source space
        if bdma_model:
            bs = 512
            logger.info("Map target embeddings to the source space ...")
            original = self.mapping.module.bidirectional
            self.mapping.module.bidirectional = True
            for i, k in enumerate(range(0, len(tgt_emb), bs)):
                x = Variable(tgt_emb[k:k + bs], volatile=True)
                fin_tgt_emb[k:k + bs] = self.mapping(x.cuda() if params.cuda else x, fdir=False).data.cpu()
            self.mapping.module.bidirectional = original

            # write embeddings to the disk
            export_embeddings(src_emb, fin_tgt_emb, params, direction="backward")
