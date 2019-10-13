import os
import sys
import time
from datetime import datetime

from PIL import Image
import numpy as np
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable

from miscc.config import cfg
from miscc.utils import build_super_images2
from model import RNN_ENCODER, G_NET

def vectorize_caption(wordtoix, caption, copies=2):
    # Create caption vector
    tokens = caption.split(' ')
    cap_v = []
    for t in tokens:
        t = t.strip().encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0 and t in wordtoix:
            cap_v.append(wordtoix[t])

    # Expected state for single generation
    captions = np.zeros((copies, len(cap_v)))
    for i in range(copies):
        captions[i,:] = np.array(cap_v)
    cap_lens = np.zeros(copies) + len(cap_v)

    return captions.astype(int), cap_lens.astype(int)

def generate(caption, wordtoix, ixtoword, text_encoder, netG, copies=2):

    # Load word vector
    captions, cap_lens  = vectorize_caption(wordtoix, caption, copies)
    n_words = len(wordtoix)

    # Only one to generate
    batch_size = captions.shape[0]

    nz = cfg.GAN.Z_DIM
    with torch.no_grad():
        captions = Variable(torch.from_numpy(captions))
        cap_lens = Variable(torch.from_numpy(cap_lens))
        noise = Variable(torch.FloatTensor(batch_size, nz))

    if cfg.CUDA:
        captions = captions.cuda()
        cap_lens = cap_lens.cuda()
        noise = noise.cuda()

    # (1) Extract text embeddings
    hidden = text_encoder.init_hidden(batch_size)
    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
    mask = (captions == 0)

    # (2) Generate fake images
    noise.data.normal_(0, 1)
    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)

    # G attention
    cap_lens_np = cap_lens.cpu().data.numpy()

    prefix = datetime.now().strftime('%Y/%B/%d/%H_%M_%S_%f')
    for j in range(batch_size):
        for k in range(len(fake_imgs)):
            im = fake_imgs[k][j].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)

            if k != len(fake_imgs) - 1:
                im.save("{}org{}_g{}.png".format(j, "bird", k))
                im = im.resize((256, 256), Image.BILINEAR)
                im.save("{}bi{}_g{}.png".format(j, "bird", k))
            else:
                im.save("{}{}_g{}.png".format(j, "bird", k))

def word_index():
    # Load word to index dictionary
    x = pickle.load(open('data/captions.pickle', 'rb'))
    ixtoword = x[2]
    wordtoix = x[3]
    del x

    return wordtoix, ixtoword

def models(word_len):
    # Create text encoder object
    text_encoder = RNN_ENCODER(word_len, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    if cfg.CUDA:
        text_encoder.cuda()
    text_encoder.eval()

    # Create generator object
    netG = G_NET()
    state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
    netG.load_state_dict(state_dict)
    if cfg.CUDA:
        netG.cuda()
    netG.eval()
    return text_encoder, netG

def main(caption):
    # load configuration
    #cfg_from_file('eval_bird.yml')

    # load word dictionaries
    wordtoix, ixtoword = word_index()

    # lead models
    text_encoder, netG = models(len(wordtoix))

    urls = generate(caption, wordtoix, ixtoword, text_encoder, netG)


if __name__ == "__main__":
    caption = "the bird has a yellow crown and a black eyering that is round"
    main(caption)
