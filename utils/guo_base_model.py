"""
voca base model version of pytorch
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, session, batcher, config, scope='defualt'):
        if 'num_render_sequences' in config:
            self.num_render_sequences = config['num_render_sequences']
        if "num_embedding_sequences" in config:
            self.num_embedding_sequences = config['num_embedding_sequences']
        if "num_embedding_samples" in config:
            self.num_embedding_sequences = config['num_embedding_samples']
        self.session = session
        self.batcher = batcher
        self.config = config
        self.scope = scope
        self.end_points = {}
        self.threads = []

        self.save_path = os.path.join(self.config['checkpoint_dir'], 'checkpoints')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _build_savers(self, max_to_keep):
        self.saver_forsave = torch.save()
        self.saver_forrestore = torch.load()

    def _save(self, step):
        save_path = os.path.join(self.config['checkpoint_dir'], 'checkpoints')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.saver_forsave(self.session, os.path.join(self.save_path, 'gstep_%s.model' % (step,)))

    def _find_closest_path(self, ckpt, approx_g_step):
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx, array[idx]

        all_paths = ckpt.all_model_checkpoint_paths
        possible_steps = []
        for checkpoint in all_paths:
            g_step = str(os.path.basename(checkpoint).split('.')[0].split('/')[-1])
            g_step = int(filter(str.isdigit, g_step))
            possible_steps.append(g_step)
        idx, chosen_g_step = find_nearest(possible_steps, approx_g_step)
        logging.warning('Loading %d step although %d was asked for' % (int(chosen_g_step), approx_g_step))
        return idx

    def load(self, epoch=None):
        if epoch is None:
            # Load latest checkpoint file
            """
            Temporally just set ckpt_name as the save path
            """
            ckpt_name = self.save_path
            if ckpt_name is not None:
                self.saver_forrestore.restore(self.session, ckpt_name)
                logging.warning("Loading model %s - this will screw up the epoch number during training" % ckpt_name)
            else:
                logging.warning("Training model from scratch")
        else:
            # approx_gstep = int(epoch * self.batcher.get_training_size() / self.config['batch_size'])
            ckpt = self.save_path
            if ckpt:
                self.saver_forrestore(self.session, ckpt)
            else:
                logging.warnings("Failed loading model")

    def _finish(self):
        for t in self.threads:
            t.join()
