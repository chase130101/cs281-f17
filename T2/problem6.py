import torch
from torch.autograd import Variable
from utils import load_imdb
from utils import bag_of_words
import numpy as np


batch_size = 100
train_iter, val_iter, test_iter, text_field = load_imdb(imdb_path='imdb.zip', imdb_dir='imdb', batch_size=batch_size, gpu=True, reuse=True, repeat=False, shuffle=True)
V = len(text_field.vocab) # vocab size
num_labels = 2

def build_model_neural(input_dim1, output_dim1, output_dim2):
    model = torch.nn.Sequential()
    # computes w_c^T x + b_c 
    model.add_module("linear1", torch.nn.Linear(input_dim1, output_dim1).cuda())
    model.add_module('tanh', torch.nn.Tanh().cuda())
    model.add_module("linear2", torch.nn.Linear(output_dim1, output_dim2).cuda())
    # Compute our log softmax term.
    model.add_module("softmax", torch.nn.LogSoftmax().cuda())
    return model
	
	
model = build_model_neural(V, int(V/1000.0), num_labels)
loss_func = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
num_epochs = 15

for epoch in range(num_epochs):
    size_training_data = 0.0
    num_correct = 0.0
    loss = 0.0
    
    for batch in train_iter:
        x = Variable(bag_of_words(batch, text_field))
        y = batch.label - 1 # batch.label is 1/2, while we want 0/1
        
        optimizer.zero_grad()
        fx = model.forward(x)
        L = loss_func(fx, y)
        L.backward()
        optimizer.step()
        
        loss += L.data[0]
        batch_num_correct = np.sum(np.argmax(torch.exp(fx).data.cpu().numpy(), axis = 1) == y.data.cpu().numpy())
        num_correct += batch_num_correct
        size_training_data += len(y)
        
    print('Epoch ' + str(epoch + 1))
    print('Epoch train accuracy: ' + str(num_correct/size_training_data))
    print('Epoch loss: ' + str(loss))
    print()
    
    
test_num_correct = 0.0
size_test_data = 0.0
for batch in test_iter:
    x = Variable(bag_of_words(batch, text_field))
    y = batch.label - 1 # batch.label is 1/2, while we want 0/1

    batch_num_correct = np.sum(np.argmax(torch.exp(model.forward(x)).data.cpu().numpy(), axis = 1) == y.data.cpu().numpy())
    test_num_correct += batch_num_correct
    size_test_data += len(y)

print('Test accuracy: ' + str(test_num_correct/size_test_data))
print()