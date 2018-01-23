#Imports
import re
import random
from time import time
from collections import Counter

import spacy
nlp = spacy.load('en')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


#Functions
def load_data(fname, min_count=2, batch_size = None):
    lines = [re.sub('[^A-Za-z0-9,]+ ', ' ' , l.strip()) for line in open(fname, 'r') for l in line.strip().split('.') if l.strip()]
    print('Loaded {} lines from file {}'.format(len(lines), fname))
    if batch_size is not None:
        lines = lines[:batch_size]
        print('Selecting {} lines for further processing'.format(batch_size))
    all_words_init = [w.lower() for line in lines for l in line.split(' ') for w in l.split(',') if w.strip()]
    vocab_init = list(set(all_words_init))
    print('Total Word Count: {}, Vocab Length: {}'.format(len(all_words_init), len(vocab_init)))

    unk_word = '<UNK>'
    unk_count = 0

    def is_unk(w, c):
        if c>= min_count:
            return w
        else:
            return unk_word
    def is_unk_cnt(w, c):
        nonlocal unk_count
        if c >= min_count:
            return c
        else:
            unk_count += c
            return unk_count

    w2f_map = {w: c for w, c in Counter(all_words_init).items() if c >= min_count}
    # all_valid_w2f_map = {}
    a = Counter(all_words_init)
    for w, c in Counter(all_words_init).items():
        if c < min_count:
            unk_count += c
    w2f_map[unk_word] = unk_count
    all_words = [w if w in w2f_map else unk_word for w in all_words_init]
    vocab = list(set(all_words + [unk_word]))
    w2i_map = {w: i for i, w in enumerate(vocab)}
    i2w_map = {i: w for w, i in w2i_map.items()}

    print('{} word count: {}'.format(unk_word, unk_count))

    return all_words, vocab, w2i_map, i2w_map, w2f_map

def generate_context_target_pairs(word_list, w2i_map, win_size, neg_sample_count=2):
    word_id_list = [w2i_map[word] for word in word_list]
    context_target_pos_word_pairs = [(word_list[tgt_ind-win_size:tgt_ind]+word_list[tgt_ind+1:tgt_ind+win_size+1], tgt_word) for tgt_ind, tgt_word in enumerate(word_list) if tgt_ind>=win_size and tgt_ind+win_size<len(word_list)]
    context_target_pos_word_id_pairs = [(word_id_list[tgt_ind-win_size:tgt_ind]+word_id_list[tgt_ind+1:tgt_ind+win_size+1], tgt_word) for tgt_ind, tgt_word in enumerate(word_id_list) if tgt_ind>=win_size and tgt_ind+win_size<len(word_id_list)]

    context_target_neg_word_ids_pairs = []
    for c_t_word_id_pair in context_target_pos_word_id_pairs:
        neg_word_ids = []
        while len(neg_word_ids)<neg_sample_count:
            neg_word_id = random.choice(word_id_list)
            if neg_word_id != c_t_word_id_pair[1] and neg_word_id not in neg_word_ids:
                neg_word_ids.append(neg_word_id)

        neg_word_ids_list = []
        for neg_word_id in neg_word_ids:
            neg_word_ids_list.append((c_t_word_id_pair[0], neg_word_id))

        context_target_neg_word_ids_pairs.append(neg_word_ids_list)
    print('Total Number of consecutive words in input file: {}'.format(len(word_id_list)))
    print('Window length in each side: {}'.format(win_size))
    print('Number of positive context-target word pairs: {}'.format(len(context_target_pos_word_id_pairs)))
    print('Number of negative samples: {}'.format(neg_sample_count))

    return context_target_pos_word_id_pairs, context_target_neg_word_ids_pairs

#Classes
class CBOW(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(CBOW, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.u_embeddings = nn.Embedding(self.vocab_size, self.emb_dim)
        self.v_embeddings = nn.Embedding(self.vocab_size, self.emb_dim)
        self.init_emb()

    def init_emb(self):
        init_range = 0.5 / self.emb_dim
        self.u_embeddings.weight.data.uniform_(-init_range, init_range)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, context_target_pos_word_id_pair = ([3, 7, 21, 42], 9), context_target_neg_word_id_pair = ([3, 7, 21, 42], 18)):
        losses = []

        pos_context_word_ids, pos_target_word_id = context_target_pos_word_id_pair
        pos_u_inp = Variable(torch.LongTensor([pos_context_word_ids]))
        pos_v_inp = Variable(torch.LongTensor([[pos_target_word_id]]))
        pos_u_embed = torch.sum(self.u_embeddings(pos_u_inp), dim=1)
        pos_v_embed = torch.sum(self.v_embeddings(pos_v_inp), dim=1)
        pos_score = F.logsigmoid(torch.sum(torch.mul(pos_u_embed, pos_v_embed), dim=1))
        losses.append(pos_score)

        neg_context_word_ids, neg_target_word_id = context_target_neg_word_id_pair
        neg_u_inp = Variable(torch.LongTensor([neg_context_word_ids]))
        neg_v_inp = Variable(torch.LongTensor([[neg_target_word_id]]))
        neg_u_embed = torch.sum(self.u_embeddings(neg_u_inp), dim=1)
        neg_v_embed = torch.sum(self.v_embeddings(neg_v_inp), dim=1)
        neg_score = F.logsigmoid(-1.0 * torch.sum(torch.mul(neg_u_embed, neg_v_embed), dim=1))
        losses.append(neg_score)

        return -1 * sum(losses)

#Functions
def save_embedding(i2w, embeds, emb_dim, fname):
    with open(fname, 'w') as fh:
        fh.write('Emb_Dim: %s\n' % str(emb_dim))
        for index, word in i2w.items():
            emb = embeds[index]
            emb = ' '.join([str(e) for e in emb])
            fh.write('%s %s\n' % (word, emb))
    pass

def CBOW_train(vocab, emb_dim, context_target_pos_word_id_pairs, context_target_neg_word_id_pairs, lrate=0.025, n_epochs=10, cuda=False):

    model = CBOW(vocab_size=len(vocab), emb_dim=emb_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=lrate)

    if cuda and torch.cuda.is_available():
        print('--> Running in GPU...')
        model = model.cuda()

    # print(len(context_target_pos_word_id_pairs))
    # print(len(context_target_neg_word_id_pairs))

    for e in range(n_epochs+1):
        t0 = time()
        losses = []
        for pos_pair, neg_pairs in zip(context_target_pos_word_id_pairs, context_target_neg_word_id_pairs):
            for neg_pair in neg_pairs:
                optimizer.zero_grad()
                loss = model.forward(pos_pair, neg_pair)
                loss.backward()
                optimizer.step()
                losses.append(loss.data[0])
                if len(losses)%50000 == 0:
                    print('Processed {}/{} word pairs in epoch {}'.format(len(losses), (len(context_target_neg_word_id_pairs)*len(context_target_neg_word_id_pairs[0])), e))
        print('Loss at epoch {}: {}, Took {} sec'.format(e, sum(losses)/len(losses), time()-t0))

    return model.u_embeddings.weight.data.numpy(), model.v_embeddings.weight.data.numpy()



#Main
if __name__ == '__main__':

    cuda = False

    input_txt_file = 'Thesis_utf8_txt.txt'
    min_count = 5
    window_size = 2
    neg_sample_count = 5
    emb_dim = 32
    lrate = 0.025
    n_epochs = 100

    all_words, vocab, w2i_map, i2w_map, w2f_map = load_data(input_txt_file, min_count)
    context_target_pos_word_id_pairs, context_target_neg_word_id_pairs = generate_context_target_pairs(all_words, w2i_map, window_size, neg_sample_count)

    u_embed, v_embed = CBOW_train(vocab, emb_dim, context_target_pos_word_id_pairs, context_target_neg_word_id_pairs, lrate, n_epochs, cuda)

    save_embedding(i2w_map, u_embed, emb_dim, input_txt_file.replace('.txt', '_u_embeds.txt'))
    save_embedding(i2w_map, v_embed, emb_dim, input_txt_file.replace('.txt', '_v_embeds.txt'))

    pass