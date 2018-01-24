#Imports
import unicodedata
import string
import re
import random
import time
import math
import numpy as np
from pprint import pprint

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

#Global Variables
use_cuda = torch.cuda.is_available()
SOS_token = 0
EOS_token = 1
MAX_LEN = 10
teacher_forcing_ratio = 0.5

#Classes
class LanguageModel:
    def __init__(self, name):
        self.name = name
        self.w2i_map = {}
        self.w2f_map = {}
        self.i2w_map = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2

    def load_sentence_to_lang_model(self, sentence):
        for word in sentence.split(' '):
            self.add_word_to_lang_model(word)

    def add_word_to_lang_model(self, word):
        if word not in self.w2i_map:
            self.w2i_map[word] = self.n_words
            self.w2f_map[word] = 1
            self.i2w_map[self.n_words] = word
            self.n_words += 1
        else:
            self.w2f_map[word] += 1

class EncoderGRU(nn.Module):
    def __init__(self, in_vocab_size, embedding_dim):
        super(EncoderGRU, self).__init__()
        self.input_dim = in_vocab_size
        self.hidden_dim = embedding_dim
        self.embedding = nn.Embedding(in_vocab_size, embedding_dim)
        self.lstm = nn.GRU(self.hidden_dim, self.hidden_dim)

    def forward(self, in_word, hidden):
        embedded_word = self.embedding(in_word).view(1, 1, -1)
        output = embedded_word
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result

class DecoderGRU(nn.Module):
    def __init__(self, embedding_dim, out_vocab_size):
        super(DecoderGRU, self).__init__()
        self.hidden_dim = embedding_dim
        self.output_dim = out_vocab_size
        self.embedding = nn.Embedding(out_vocab_size, embedding_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, in_embed, hidden):
        output = self.embedding(in_embed).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.log_softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result

class AttentionDecoderGRU(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout_p=0.1, max_length=MAX_LEN):
        super(AttentionDecoderGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_p = dropout_p
        self.max_len = max_length
        self.embedding = nn.Embedding(self.output_dim, self.hidden_dim)
        self.attention = nn.Linear(self.hidden_dim*2, self.max_len)
        self.attention_combine = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, inp, hidden, encoder_outputs):
        embedded = self.embedding(inp).view(1, 1, -1)
        embedded = self.dropout(embedded)
        # attention_weights = F.softmax(self.attention(torch.cat(embedded[0], hidden[0], 1)), dim=1)
        attention_weights = F.softmax(self.attention(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attention_applied = torch.bmm(attention_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attention_applied[0]), 1)
        output = self.attention_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attention_weights

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result


#Functions
def as_minutes(s):
    m = math.floor(s/60)
    s -= m*60
    return '{}min and {}sec'.format(m, s)

def time_since(since, percent):
    now = time.time()
    s = now-since
    es = s/percent
    rs = es - s
    return '{} (- {})'.format(as_minutes(s), as_minutes(rs))

def normalize_word(word):
    word = word.lower().strip()
    word = re.sub(r"([.!?}])", r" \1", word)
    # word = re.sub(r"[^a-zA-Z.!?]+", r" ", word)
    return word

def load_lang_model_from_file(fname, reverse=False, normalize_lang1=False, normalize_lang2=False):
    print('Reading Lines from file: {}'.format(fname))
    lines = open(fname, encoding='utf-8').read().strip().split('\n')
    # print('{} lines are read'.format(len(lines)))
    pairs = [[normalize_word(word) for word in line.split('\t')] for line in lines]
    input_lang, output_lang = fname.replace('.txt', '').split('-')
    print('Identified languages: <{}> and <{}>'.format(input_lang, output_lang))

    # if normalize_lang1:
    #     pairs = [[' '.join([normalize_word(word) for word in pair[0].split(' ')]), pair[1]] for pair in pairs]
    # if normalize_lang2:
    #     pairs = [[pair[0], ' '.join([normalize_word(word) for word in pair[1].split(' ')])] for pair in pairs]

    if reverse:
        pairs = [list(reversed(pair)) for pair in pairs]
        src_lang = LanguageModel(output_lang)
        tgt_lang = LanguageModel(input_lang)
        print('Reverse translation')
    else:
        src_lang = LanguageModel(input_lang)
        tgt_lang = LanguageModel(output_lang)
        print('Forward translation')
    print('Source language: <{}>, Target language: <{}>'.format(src_lang.name, tgt_lang.name))
    return src_lang, tgt_lang, pairs

def filter_pairs(pairs, max_len=1e4):
    return [pair for pair in pairs if len(pair[0].split(' '))<max_len and len(pair[1].split(' '))<max_len]

def prepare_data(fname, reverse=False):
    src_lang, tgt_lang, pairs = load_lang_model_from_file(fname, reverse, normalize_lang1=True, normalize_lang2=True)
    print('Loaded {} sentences pairs'.format(len(pairs)))
    pairs = filter_pairs(pairs, max_len=MAX_LEN)
    print('Filtering the pairs based on MAX_LEN: {}'.format(MAX_LEN))
    print('Selected {} sentences pairs after filtering'.format(len(pairs)))
    print('Counting words...')
    for pair in pairs:
        src_lang.load_sentence_to_lang_model(pair[0])
        tgt_lang.load_sentence_to_lang_model(pair[1])
    # print('Counted {} words in <{}>\nCounted {} words in <{}>'.format(src_lang.n_words, src_lang.name, tgt_lang.n_words, tgt_lang.name))
    print('Vocab size of <{}> (including SOS, EOS): {}\nVocab size of <{}> (including SOS, EOS): {}'.format(src_lang.name, src_lang.n_words, tgt_lang.name, tgt_lang.n_words))
    return src_lang, tgt_lang, pairs

def indexes_from_sentence(lang, sentence):
    return [lang.w2i_map[word] for word in sentence.split(' ')]

def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

def variables_from_pair(pair):
    src_variable = variable_from_sentence(src_lang, pair[0])
    tgt_variable = variable_from_sentence(tgt_lang, pair[1])
    return (src_variable, tgt_variable)

def train(src_variable, tgt_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_len=MAX_LEN):
    encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    src_len = src_variable.size()[0]
    tgt_len = tgt_variable.size()[0]
    encoder_outputs = Variable(torch.zeros(max_len, encoder.hidden_dim))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(src_len):
        encoder_output, encoder_hidden = encoder(src_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        #Teacher forcing: use the target as the next input
        for di in range(tgt_len):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, tgt_variable[di])
            decoder_input = tgt_variable[di] #Teacher forcing
    else:
        #No teacher forcing: use its own predictions as the next input
        for di in range(tgt_len):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            loss += criterion(decoder_output, tgt_variable[di])
            if ni == EOS_token:
                break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.data[0]/tgt_len

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def train_iters(encoder, decoder, n_iters, print_every=1000, plot_every=100, lrate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=lrate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=lrate)
    training_pairs = [variables_from_pair(random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()
    for iter in range(1, n_iters+1):
        training_pair = training_pairs[iter-1]
        src_variable = training_pair[0]
        tgt_variable = training_pair[1]
        loss = train(src_variable, tgt_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        if iter%print_every == 0:
            print_loss_avg = print_loss_total/print_every
            print_loss_total = 0
            print('Took {} (Iteration:{}, Completed {}%%) Average Loss: {}'.format(time_since(start, iter/n_iters), iter, iter/n_iters*100, print_loss_avg))
        if iter%plot_every == 0:
            plot_loss_avg = plot_loss_total/plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    show_plot(plot_losses)

def evaluate(encoder, decoder, sentence, max_len=MAX_LEN):
    src_variable = variable_from_sentence(src_lang, sentence)
    src_len = src_variable.size()[0]
    encoder_hidden = encoder.init_hidden()
    encoder_outputs = Variable(torch.zeros(max_len, encoder.hidden_dim))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    for ei in range(src_len):
        encoder_output, encoder_hidden = encoder(src_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden
    decoded_words = []
    decoder_attentions = torch.zeros(max_len, max_len)
    for di in range(max_len):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(tgt_lang.i2w_map[ni])
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    return decoded_words

def evaluate_randomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


#Main
if __name__ == '__main__':
    fname = 'eng-tam.txt'
    src_lang, tgt_lang, pairs = prepare_data(fname)
    pprint(random.choice(pairs))
    hidden_size = 256
    encoder1 = EncoderGRU(src_lang.n_words, hidden_size)
    attn_decoder1 = AttentionDecoderGRU(hidden_size, tgt_lang.n_words, dropout_p=0.1)
    if use_cuda:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()
    train_iters(encoder1, attn_decoder1, 25000, print_every=5000)
    evaluate_randomly(encoder1, attn_decoder1)