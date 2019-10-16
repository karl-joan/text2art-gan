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

from .config import cfg, cfg_from_file
from .model import RNN_ENCODER, G_NET

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

def generate(caption, wordtoix, ixtoword, text_encoder, netG, dataset, copies=2):

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

    # Make a save directory and change the current direcoty to it
    mydir = os.path.join("../results/", datetime.today().strftime("%Y-%m-%d_%H-%M-%S/"))
    try:
        os.makedirs(mydir)
        prefix = mydir
    except:
        prefix = ""
        print("Failed to create a save directory")

    for j in range(batch_size):
        for k in range(len(fake_imgs)):
            im = fake_imgs[k][j].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)

            if k != len(fake_imgs) - 1:
                im = im.resize((256, 256), Image.BILINEAR)
                im.save(prefix + "{}{}_g{}.png".format(j, dataset, k))
            else:
                im.save(prefix + "{}{}_g{}.png".format(j, dataset, k))

def word_index():
    # Load word to index dictionary
    x = pickle.load(open(cfg.TEXT.CAPTIONS, 'rb'))
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

def attngan(caption, dataset):
    # Choose the model
    if dataset == "birds":
        cfg_from_file("attngan/cfg/eval_bird.yml")
    else: # dataset == "coco"
        cfg_from_file("attngan/cfg/eval_coco.yml")

    # Load word dictionaries
    wordtoix, ixtoword = word_index()

    # Lead models
    text_encoder, netG = models(len(wordtoix))

    # Generate images
    generate(caption, wordtoix, ixtoword, text_encoder, netG, dataset)


if __name__ == "__main__":
    caption = "the bird has a yellow crown and a black eyering that is round"
    attngan(caption, "birds")
    #caption = "a fruit stand display with bananas and kiwi"
    #attngan(caption, "coco")
