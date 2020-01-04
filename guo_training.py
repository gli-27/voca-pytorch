import os
import torch
import ConfigParser
import shutil
import logging
from torch.utils.data import DataLoader
from utils.audio_loader import DataHandler, Batcher
from config_parser import read_config, create_default_config
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.guo_model import VOCAModel

def one_hot(x):
    x = x.unsqueeze(-1)
    condition = torch.zeros(x.shape[0], 8).scatter_(1, x.type(torch.LongTensor), 1)
    return condition

def main():
    pkg_path, _ = os.path.split(os.path.realpath(__file__))
    init_config_fname = os.path.join(pkg_path, 'training_config.cfg')
    if not os.path.exists(init_config_fname):
        print('Config not found %s' % init_config_fname)
        create_default_config(init_config_fname)

    config = ConfigParser.RawConfigParser()
    config.read(init_config_fname)

    # Path to cache the processed audio
    config.set('Input Output', 'processed_audio_path',
               './training_data/processed_audio_%s.pkl' % config.get('Audio Parameters', 'audio_feature_type'))

    checkpoint_dir = config.get('Input Output', 'checkpoint_dir')
    if os.path.exists(checkpoint_dir):
        print('Checkpoint dir already exists %s' % checkpoint_dir)
        key = raw_input('Press "q" to quit, "x" to erase existing folder, and any other key to continue training: ')
        if key.lower() == 'q':
            return
        elif key.lower() == 'x':
            try:
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
            except:
                print('Failed deleting checkpoint directory')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    config_fname = os.path.join(checkpoint_dir, 'config.pkl')
    if os.path.exists(config_fname):
        print('Use existing config %s' % config_fname)
    else:
        with open(config_fname, 'wb') as fp:
            config.write(fp)
            fp.close()

    config = read_config(config_fname)

    data_handler = DataHandler(config)
    batcher = Batcher(data_handler)
    loader = DataLoader(batcher, batch_size=config['batch_size'], num_workers=8, shuffle=True)

    world_size = 1
    rank = 0
    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

    model = VOCAModel(config)
    model = model.to(device_ids[0])

    for epoch in range(1, config['epoch_num']+1):
        num_train_batches = batcher.get_num_batches(config['batch_size'])
        for iter in range(num_train_batches):
            loss = _training_step(config, model, batcher, device_ids)
            if iter % 50 == 0:
                logging.warning("Epoch: %d | Iter: %d | Loss: %.6f | Learning Rate: %.6f" % (
                epoch, iter, loss, config['learning_rate']))
            if iter % 100 == 0:
                valid_loss = _validation_step(config, model, batcher, device_ids)
                logging.warning("Validation loss: %.6f" % valid_loss)



def _training_step(config, model, batcher, device_ids):
    model.optim_netE.zero_grad()
    model.optim_netD.zero_grad()
    processed_audio, face_vertices, face_templates, subject_idx = batcher.get_training_batch(config['batch_size'])
    processed_audio = processed_audio.to(device_ids[0])
    face_vertices = face_vertices.to(device_ids[0])
    face_templates = face_templates.to(device_ids[0])
    subject_idx = subject_idx
    condition = one_hot(subject_idx)
    condition = condition.to(device_ids[0])

    feat = model.speech_encoder(processed_audio, condition)
    expression_offset = model.decoder(feat)
    predict = expression_offset + face_templates

    recon_loss = model.recon_loss(predict, face_vertices)
    velocity_loss = model.velocity_loss(predict, face_vertices)
    acc_loss = model.acc_loss(predict, face_vertices)
    vert_reg_loss = model.verts_reg_loss(expression_offset)
    loss = recon_loss + velocity_loss + acc_loss + vert_reg_loss
    loss.backward()
    model.optim_netE.step()
    model.optim_netD.step()
    return loss

def _validation_step(config, model, batcher, device_ids):
    processed_audio, face_vertices, face_templates, subject_idx = batcher.get_training_batch(config['batch_size'])
    processed_audio = processed_audio.to(device_ids[0])
    face_vertices = face_vertices.to(device_ids[0])
    face_templates = face_templates.to(device_ids[0])
    subject_idx = subject_idx
    condition = one_hot(subject_idx)
    condition = condition.to(device_ids[0])

    feat = model.speech_encoder(processed_audio.detach(), condition.detach())
    expression_offset = model.decoder(feat.detach())
    predict = expression_offset + face_templates

    recon_loss = model.recon_loss(predict, face_vertices)
    velocity_loss = model.velocity_loss(predict, face_vertices)
    acc_loss = model.acc_loss(predict, face_vertices)
    vert_reg_loss = model.verts_reg_loss(expression_offset)
    loss = recon_loss + velocity_loss + acc_loss + vert_reg_loss
    return loss

if __name__ == '__main__':
    main()