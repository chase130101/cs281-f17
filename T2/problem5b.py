import torch
from torch.autograd import Variable
from utils import load_imdb
from utils import bag_of_words
import numpy as np


batch_size = 100
train_iter, val_iter, test_iter, text_field = load_imdb(imdb_path='imdb.zip', imdb_dir='imdb', batch_size=batch_size, gpu=True, reuse=True, repeat=False, shuffle=True)
V = len(text_field.vocab) # vocab size
num_labels = 2
vocab_list = text_field.vocab.itos


def build_model(input_dim, output_dim):
    model = torch.nn.Sequential()
    # computes w_c^T x + b_c 
    model.add_module("linear", torch.nn.Linear(input_dim, output_dim).cuda())
    # Compute our log softmax term.
    model.add_module("softmax", torch.nn.LogSoftmax().cuda())
    return model


def l1_logistic_loss(model, lambda_, fx, y):
    log_loss = torch.nn.NLLLoss(size_average = True)
    log_loss = log_loss.forward(fx, y)
    
    lasso_part = torch.nn.L1Loss(size_average = False)
    params = next(model.parameters())
    target = Variable(torch.zeros(params.size()[0], params.size()[1]).cuda(), requires_grad=False)
    lasso_part = lasso_part.forward(params, target)
    
    return log_loss + lasso_part * lambda_
    
    
def train(model, lambda_, x, y, optimizer):
    # Resets the gradients to 0
    optimizer.zero_grad()
    # Computes the function above. (log softmax w_c^T x + b_c)
    fx = model.forward(x)
    # Computes a loss. Gives a scalar. 
    loss = l1_logistic_loss(model, lambda_, fx, y)
    # Magically computes the gradients. 
    loss.backward()
    # updates the weights
    optimizer.step()
    return loss.data[0]
	
	
lam_vals = [0, 0.001, 0.01, 0.1, 1]
num_epochs = 15

for lam in lam_vals:
    model = build_model(V, num_labels)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    size_val_data = 0.0
    val_num_correct = 0.0

    for epoch in range(num_epochs):
        size_training_data = 0.0
        train_num_correct = 0.0
        loss = 0.0

        for batch in train_iter:
            x = Variable(bag_of_words(batch, text_field))
            y = batch.label - 1 # batch.label is 1/2, while we want 0/1

            batch_loss = train(model, lam, x, y, optimizer)
            loss += batch_loss

            batch_num_correct = np.sum(np.argmax(torch.exp(model.forward(x)).data.cpu().numpy(), axis = 1) == y.data.cpu().numpy())
            train_num_correct += batch_num_correct
            size_training_data += len(y)
        
        print('Epoch ' + str(epoch + 1))
        print('Lambda = ' + str(lam))
        print('Epoch train accuracy: ' + str(train_num_correct/size_training_data))
        print('Epoch train loss: ' + str(loss))
        print()
        
    
    for batch in val_iter:
        x = Variable(bag_of_words(batch, text_field))
        y = batch.label - 1 # batch.label is 1/2, while we want 0/1
        
        batch_num_correct = np.sum(np.argmax(torch.exp(model.forward(x)).data.cpu().numpy(), axis = 1) == y.data.cpu().numpy())
        val_num_correct += batch_num_correct
        size_val_data += len(y)
        
    print('Lambda = ' + str(lam))
    print('Val accuracy: ' + str(val_num_correct/size_val_data))
    print()

    
    test_num_correct = 0.0
    size_test_data = 0.0
    for batch in test_iter:
        x = Variable(bag_of_words(batch, text_field))
        y = batch.label - 1 # batch.label is 1/2, while we want 0/1

        batch_num_correct = np.sum(np.argmax(torch.exp(model.forward(x)).data.cpu().numpy(), axis = 1) == y.data.cpu().numpy())
        test_num_correct += batch_num_correct
        size_test_data += len(y)

    print('Lambda: ' + str(lam))
    print('Test accuracy: ' + str(test_num_correct/size_test_data))
    print()

    model_params = next(model.parameters())
    print('Words with highest valued coefficients for predicting class 0: ' + str(np.flip(np.array(vocab_list)[np.argsort(model_params[0].data.cpu().numpy())[-5:]], axis = 0)))
    print('Words with lowest valued coefficients for predicting class 0: ' + str(np.array(vocab_list)[np.argsort(model_params[0].data.cpu().numpy())[0:5]]))
    print('Words with highest valued coefficients for predicting class 1: ' + str(np.flip(np.array(vocab_list)[[np.argsort(model_params[1].data.cpu().numpy())[-5:]]], axis = 0)))
    print('Words with lowest valued coefficients for predicting class 1: ' + str(np.array(vocab_list)[[np.argsort(model_params[1].data.cpu().numpy())[0:5]]]))
    print()
	
    for j in range(num_labels):
        abs_params = np.absolute(model_params.data.cpu().numpy()[j])
        print('Number of 0 valued parameters for class ' + str(j) + ' (lambda=' + str(lam) + '): ' + str(len(abs_params[abs_params < 10.0**(-4)])/V))
    print()	
	
	
	
	
	
	
