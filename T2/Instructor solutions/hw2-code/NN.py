import numpy as np
import scipy.stats
import torch
from torch import optim
from torch.autograd import Variable
import string
import re
from utils import *

torch.manual_seed(42)

def build_model(input_dim, hidden_dim, output_dim):
    # We don't need the softmax layer here since CrossEntropyLoss already
    # uses it internally.
    model = torch.nn.Sequential()
    model.add_module("linear",
    torch.nn.Linear(input_dim, hidden_dim, bias=False))
    model.add_module("relu",
    torch.nn.ReLU())
    model.add_module("linear2",
    torch.nn.Linear(hidden_dim, output_dim, bias=False))
    return model


def train_(model, loss, optimizer, x, y):
    # Reset gradient
    optimizer.zero_grad()
    # Forward
    fx = model.forward(x)
    output = loss.forward(fx, y)
    # Backward
    output.backward()
    # Update parameters
    optimizer.step()
    return output.data[0]

def predict_(model, x):
    output = model.forward(x)
    return output.data.numpy().argmax(axis=1)


def main():
    hidden_dim = 50
    batch_size = 100
    num_epochs = 10
    report_every = 24
    model_path = 'model'

    train_iter, val_iter, test_iter, text_field = load_imdb(batch_size=batch_size)
    V = len(text_field.vocab)

    model = build_model(V, hidden_dim, 2)
    loss = torch.nn.CrossEntropyLoss(size_average=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    cost = 0.
    best_accu = -1
    for epoch in range(num_epochs):
        print ('Starting epoch %d'%(epoch+1))
        for batch in train_iter:
            x = bag_of_words(batch, text_field)
            y = batch.label - 1
            cost += train_(model, loss, optimizer, x, y)
            iterations = train_iter.iterations
            if iterations%report_every==0 and iterations>0:
                print ('Epoch %d %.2f%%. Loss %f'%(epoch+1, 100*train_iter.epoch, cost/report_every))
                cost = 0
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

if __name__ == "__main__":
    main()
