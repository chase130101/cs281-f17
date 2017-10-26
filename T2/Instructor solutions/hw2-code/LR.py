import numpy as np
import time
import scipy.stats
import torch
from torch import optim
from torch.autograd import Variable
import string
import re
from utils import *

torch.manual_seed(42)

def build_model(input_dim, output_dim):
    # We don't need the softmax layer here since CrossEntropyLoss already
    # uses it internally.
    model = torch.nn.Sequential()
    model.add_module("linear",
    torch.nn.Linear(input_dim, output_dim, bias=False))
    return model


def train_(model, loss, optimizer, C, x, y):
    # Reset gradient
    optimizer.zero_grad()
    # Forward
    fx = model.forward(x)
    output = loss.forward(fx, y)
    for param in model.parameters():
        p = param[0] - param[1]
        output += C*p.abs().sum()
    # Backward
    output.backward()
    # Update parameters
    optimizer.step()
    return output.data[0]

def predict_(model, x):
    output = model.forward(x)
    return output.data.numpy().argmax(axis=1)


def main(C, train_iter, val_iter, test_iter, num_epochs):
    report_every = 24
    model_path = 'model'

    V = len(text_field.vocab)

    model = build_model(V, 2)
    loss = torch.nn.CrossEntropyLoss(size_average=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    cost = 0.
    best_accu = -1
    for epoch in range(num_epochs):
        print ('Starting epoch %d'%(epoch+1))
        start = time.time()
        for batch in train_iter:
            x = bag_of_words(batch, text_field)
            y = batch.label - 1
            cost += train_(model, loss, optimizer, C, x, y)
            iterations = train_iter.iterations
            if iterations%report_every==0 and iterations>0:
                print ('Epoch %d %.2f%%. Loss %f'%(epoch+1, 100*train_iter.epoch, cost/report_every))
                cost = 0
        end = time.time()
        print (end-start)
        # Validate after end of each epoch
        print ('Validating')
        correct, total = 0., 0
        for batch in val_iter:
            x = bag_of_words(batch, text_field)
            y = batch.label - 1
            y_pred = predict_(model, x)
            correct += np.sum(y_pred==y.data.numpy())
            total += y_pred.shape[0]
        accuracy = correct/total*100
        print ('Epoch %d. Accuracy %.2f%%'%(epoch, accuracy))
        if accuracy > best_accu:
            best_accu = accuracy
            print ('Best Model so far, saving...')
            torch.save(model.state_dict(), model_path)
        train_iter.init_epoch()

    # Test using the best validation model.
    print ('Testing')
    model.load_state_dict(torch.load(model_path))
    for batch in test_iter:
        x = bag_of_words(batch, text_field)
        y = batch.label - 1
        y_pred = predict_(model, x)
        correct += np.sum(y_pred==y.data.numpy())
        total += y_pred.shape[0]
    accuracy = correct/total*100
    print ('Final Test Accuracy %.2f%%'%(accuracy))

    # Most positive and most negative words
    parameters = []
    for param in model.parameters():
        parameters.append(param)
    print (len(parameters))
    parameter = parameters[0].data.numpy()
    p = parameter[0] - parameter[1]
    print (p.shape)
    inds = np.argsort(p)
    print ('Most positive words')
    for i in range(5):
        ind = inds[i]
        print (p[ind])
        word = text_field.vocab.itos[ind]
        print (word)
    print ('Most negative words')
    for i in range(5):
        ind = inds[-(1+i)]
        print (p[ind])
        word = text_field.vocab.itos[ind]
        print (word)

    print (p)
    num_zeros = np.sum(np.fabs(p) < 1e-4)
    sparsity = float(num_zeros) / p.shape[0]
    print ('Sparsity: %f'%sparsity)

if __name__ == "__main__":
    batch_size = 100
    num_epochs = 10
    train_iter, val_iter, test_iter, text_field = load_imdb(batch_size=batch_size)
    for C in [0, 0.001, 0.01, 0.1, 1]:
        print ('-----------------')
        print ('C: %f'%C)
        main(C=C, train_iter=train_iter, val_iter=val_iter, test_iter=test_iter, num_epochs=num_epochs)
