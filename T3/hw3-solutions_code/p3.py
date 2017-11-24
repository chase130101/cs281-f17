import sys, os, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchtext
from torchtext import data
from utils import *
torch.manual_seed(1236)



class Model(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super(Model, self).__init__()
        inf = 100# float('inf')
        self.b_r = torch.Tensor([-4, -2, 0, 2, 4])
        self.b_r_plus_1 = torch.Tensor([-2, 0, 2, 4, inf])

        self.w = nn.Embedding(vocab_size, 1, padding_idx=padding_idx)
        torch.nn.init.uniform(self.w.weight.data, -0.01,0.01)

        self.log_sigma = nn.Parameter(torch.FloatTensor(1))
        
    def forward(self, text, rating):
        h = torch.squeeze(torch.sum(self.w(text), dim=1))
        self.sigma = torch.exp(self.log_sigma)
        b_r = Variable(self.b_r.index(rating.data))
        b_r_plus_1 = Variable(self.b_r_plus_1.index(rating.data))
        log_p_comb = log_difference((h-b_r)/self.sigma, (h-b_r_plus_1)/self.sigma)

        return (-torch.sum(log_p_comb)/100+0.5*torch.sum(self.w.weight*self.w.weight), log_p_comb)


def val(val_iter, model):
    val_iter.init_epoch()
    total_error = 0.
    total_num = 0
    for batch in val_iter:
        text = batch.text[0] # x is a tensor of size batch_size x max_len, where max_len
        total_num = total_num + text.size(0)
        true_rating = batch.ratings-1 # batch.label is a tensor containing actual ratings 1/2/3/4/5.
        rating = torch.LongTensor(text.data.numpy().shape[0])
        results = []
        for r in range(5):
            rating.fill_(r)
            _, loss = model.forward(text, Variable(rating))
            results.append(loss.data.numpy().tolist())
        results = np.array(results) #batch_size*5
        predictions = np.argmax(results, axis=0)
        total_error += np.sum(np.square(predictions-true_rating.data.numpy()))
    return math.sqrt(total_error/total_num)


def train(train_iter, val_iter, test_iter, model):
    opt = optim.Adam(model.parameters(), lr=0.0001)

    print("val:", val(val_iter, model))
    for epochs in range(50):
        avg_loss = 0
        total = 0
        train_iter.init_epoch()
        for i,batch in enumerate(train_iter):
            opt.zero_grad()
            text = batch.text[0] # x is a tensor of size batch_size x max_len, where max_len
            rating = batch.ratings-1 # batch.label is a tensor containing actual ratings 1/2/3/4/5.
            loss, _ = model.forward(text, rating)
            if i % 1000==0:
                print (i*100, loss.data[0])
            loss.backward()
            avg_loss += loss.data[0]
            total += 1
            opt.step()
        print("train:", avg_loss / float(total))
        print("val:", val(val_iter, model))

      
batch_size = 100
train_iter, val_iter, test_iter, text_field = load_jester(batch_size=batch_size, subsample_rate=0.2)
vocab_size = len(text_field.vocab) # vocab size
print ('vocab_size', vocab_size)
model = Model(vocab_size=vocab_size, padding_idx=text_field.vocab.stoi['<pad>'])
train(train_iter, val_iter, test_iter, model)
