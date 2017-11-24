import torch
import gzip
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import *
import math


def read_data():
    with gzip.open("jester_ratings.dat.gz") as f:
        X = []
        y = []
        for i, l in enumerate(f):
            user, joke, rating = l.split()
            X.append([int(user), int(joke)])
            y.append(float(rating))
        
    perm = torch.randperm(len(X))
    X = torch.LongTensor(X).index(perm)
    y = torch.FloatTensor(y).index(perm)
    return X, y


class Model(nn.Module):
    def __init__(self, k=2):
        super(Model, self).__init__()
        self.user_lut = nn.Embedding(63979, k)
        self.joke_lut = nn.Embedding(151, k)

        self.user_bias = nn.Embedding(63979, 1)
        self.joke_bias = nn.Embedding(151, 1)
        self.global_bias = nn.Parameter(torch.FloatTensor(1))
        
    def forward(self, users, jokes):
        user_vectors = self.user_lut(users)
        joke_vectors = self.joke_lut(jokes)
        user_bias = self.user_bias(users)
        joke_bias = self.joke_bias(jokes)

        return torch.bmm(user_vectors.unsqueeze(1),
                         joke_vectors.unsqueeze(2)).squeeze() \
                         + user_bias.squeeze() + joke_bias.squeeze() + self.global_bias.expand_as(user_bias.squeeze())


def val(val_iter, model):
    val_iter.init_epoch()
    crit = nn.MSELoss(size_average=False)
    total_loss = 0.
    total_num = 0
    for batch in val_iter:
        true_rating = batch.ratings.float()-1 # batch.label is a tensor containing actual ratings 1/2/3/4/5.
        total_num = total_num + true_rating.size(0)
        users = batch.users
        jokes = batch.jokes
        scores = model.forward(users, jokes)
        total_loss += crit(scores, true_rating).data[0]
    return math.sqrt(total_loss/total_num)



def train(train_iter, val_iter, test_iter, model):
    opt = optim.SGD(model.parameters(), lr=0.1)
    crit = nn.MSELoss()

    print("val:", val(val_iter, model))
    for epochs in range(10):
        avg_loss = 0
        total = 0
        train_iter.init_epoch()
        for i,batch in enumerate(train_iter):
            opt.zero_grad()
            rating = batch.ratings.float()-1 # batch.label is a tensor containing actual ratings 1/2/3/4/5.
            users = batch.users
            jokes = batch.jokes
            scores = model.forward(users, jokes)
            loss = crit(scores, rating)
            #if i % 1000==0:
            #    print (loss.data[0])
            loss.backward()
            avg_loss += loss.data[0]
            total += 1
            opt.step()
        print("train:", math.sqrt(avg_loss / float(total)))
        print("val:", val(val_iter, model))
      
batch_size = 100
train_iter, val_iter, test_iter = load_jester(batch_size=batch_size, load_text=False)
model = Model(k=2)
train(train_iter, val_iter, test_iter, model)
