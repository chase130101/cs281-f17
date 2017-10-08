import torch
from torch.autograd import Variable
from utilsCPU import load_imdbCPU
from utilsCPU import bag_of_wordsCPU
import numpy as np


batch_size = 3000
train_iter, val_iter, test_iter, text_field = load_imdbCPU(imdb_path='imdb.zip', imdb_dir='imdb', batch_size=batch_size, gpu=False, reuse=True, repeat=False, shuffle=True)
V = len(text_field.vocab)
num_labels = 2


def NB_train(train_iter, class_labels_dir_prior_params, vocab_length, num_labels, beta0_prior_params=None, beta1_prior_params=None, dir0_unif_prior_param=None, dir1_unif_prior_param=None, maxD=None,\
             betaPrior=True):
    class_labels_dir_posterior_params = torch.addcmul(torch.zeros(num_labels).double(), value=1, tensor1=torch.ones(num_labels).double(), \
                                                      tensor2=class_labels_dir_prior_params)
    if betaPrior == True:
        beta00_posterior_vec = torch.ones(vocab_length).double()*beta0_prior_params[0]
        beta10_posterior_vec = torch.ones(vocab_length).double()*beta0_prior_params[1]
        beta01_posterior_vec = torch.ones(vocab_length).double()*beta1_prior_params[0]
        beta11_posterior_vec = torch.ones(vocab_length).double()*beta1_prior_params[1]
        for batch in train_iter:
            x = bag_of_wordsCPU(batch, text_field).double()
            y = torch.from_numpy(torch.unsqueeze((batch.label - 1).double(), 1).data.numpy()).double()

            x = x.div(x)
            x[x != x] = 0
            
            class_labels_dir_posterior_params.add_(torch.DoubleTensor([sum(1 - y).numpy(), sum(y).numpy()]))
            #beta00_posterior_vec.add_(torch.matmul(torch.transpose(1 - x, 0, 1), 1 - y))
            #beta10_posterior_vec.add_(torch.matmul(torch.transpose(x, 0, 1), 1 - y))
            #beta01_posterior_vec.add_(torch.matmul(torch.transpose(1 - x, 0, 1), y))
            #beta11_posterior_vec.add_(torch.matmul(torch.transpose(x, 0, 1), y))
            for i in range(len(y)):
                if y[i].numpy() == 0:
                    beta00_posterior_vec.add_(1 - x[i])
                    beta10_posterior_vec.add_(x[i])

                if y[i].numpy() == 1:
                    beta01_posterior_vec.add_(1 - x[i])
                    beta11_posterior_vec.add_(x[i])

        class_label_dir_posterior_expectation = class_labels_dir_posterior_params/sum(class_labels_dir_posterior_params)
        beta0_posterior_expectation = beta10_posterior_vec/(beta00_posterior_vec + beta10_posterior_vec)
        beta1_posterior_expectation = beta11_posterior_vec/(beta01_posterior_vec + beta11_posterior_vec)

        return class_label_dir_posterior_expectation, beta0_posterior_expectation, beta1_posterior_expectation
    
    
    if betaPrior == False:
        
        count = 0
        dir0_posterior_matrix = np.zeros((maxD + 1, vocab_length)) + dir0_unif_prior_param
        dir1_posterior_matrix = np.zeros((maxD + 1, vocab_length)) + dir1_unif_prior_param
        
        for batch in train_iter:
            
            #print(count, end = ' ')
            #count += 1
                
            x = bag_of_wordsCPU(batch, text_field).numpy()
            y = torch.from_numpy(torch.unsqueeze((batch.label - 1).double(), 1).data.numpy()).double()
        
            class_labels_dir_posterior_params.add_(torch.DoubleTensor([sum(1 - y).numpy(), sum(y).numpy()]))
            
            y = (batch.label - 1).data.numpy()
            x0 = x[np.where(y == 0), :][0]
            x1 = x[np.where(y == 1), :][0]
            
            for i in range(x.shape[1]):

                vals0 =  np.unique(x0[:, i], return_counts = True)[0]
                counts0 = np.unique(x0[:, i], return_counts = True)[1]
                vals1 =  np.unique(x1[:, i], return_counts = True)[0]
                counts1 = np.unique(x1[:, i], return_counts = True)[1]

                vals0_maxD = vals0[np.where(vals0 < maxD)].tolist()
                counts0_maxD = counts0[np.where(vals0 < maxD)].tolist()
                vals1_maxD = vals1[np.where(vals1 < maxD)].tolist()
                counts1_maxD = counts1[np.where(vals1 < maxD)].tolist()

                vals0_maxD.append(maxD)
                counts0_maxD.append(np.sum(counts0[np.where(vals0 >= maxD)]))
                vals1_maxD.append(maxD)
                counts1_maxD.append(np.sum(counts1[np.where(vals1 >= maxD)]))
                
                dir0_posterior_matrix[:, i][np.array(vals0_maxD).astype(int)] += np.array(counts0_maxD)
                dir1_posterior_matrix[:, i][np.array(vals1_maxD).astype(int)] += np.array(counts1_maxD)
        
        class_label_dir_posterior_expectation = class_labels_dir_posterior_params/sum(class_labels_dir_posterior_params)
        dir0_posterior_expectation = torch.from_numpy(np.divide(dir0_posterior_matrix, np.sum(dir0_posterior_matrix, axis = 0))).double()
        dir1_posterior_expectation = torch.from_numpy(np.divide(dir1_posterior_matrix, np.sum(dir1_posterior_matrix, axis = 0))).double()
        
        return class_label_dir_posterior_expectation, dir0_posterior_expectation, dir1_posterior_expectation
		
		
def compute_accuracy(dataset, vocab_length, class_label_dir_posterior_expectation, beta0_posterior_expectation=None, beta1_posterior_expectation=None, dir0_posterior_expectation=None, \
                     dir1_posterior_expectation=None, maxD=None, betaPrior = True):
    test_num_correct = torch.zeros(1).double()
    size_test_data = torch.zeros(1).double()
    if betaPrior == True:
        for batch in dataset:
            x = bag_of_wordsCPU(batch, text_field).double()
            y = torch.from_numpy((batch.label - 1).double().data.numpy()).double() # batch.label is 1/2, while we want 0/1

            xi1_ind_class0 = torch.matmul(x, torch.log(beta0_posterior_expectation))
            xi0_ind_class0 = torch.matmul(1.0 - x, torch.log(1.0 - beta0_posterior_expectation))
            log_likelihood_class0 = torch.unsqueeze(torch.log(class_label_dir_posterior_expectation)[0] + xi1_ind_class0 + xi1_ind_class0, 1)

            xi1_ind_class1 = torch.matmul(x, torch.log(beta1_posterior_expectation))
            xi0_ind_class1 = torch.matmul(1.0 - x, torch.log(1.0 - beta1_posterior_expectation))
            log_likelhood_class1 = torch.unsqueeze(torch.log(class_label_dir_posterior_expectation)[1] + xi1_ind_class1 + xi1_ind_class1, 1)

            test_num_correct.add_(sum(torch.from_numpy(np.argmax(torch.cat([log_likelihood_class0, \
                                                                            log_likelhood_class1], 1).numpy(), axis = 1)).double() == y))
            size_test_data.add_(len(y))

        return test_num_correct/size_test_data
    
    if betaPrior == False:
        
        log_dir0_posterior_expectation = np.log(dir0_posterior_expectation.numpy())
        log_dir1_posterior_expectation = np.log(dir1_posterior_expectation.numpy())
        
        for batch in dataset:
            x = bag_of_wordsCPU(batch, text_field).numpy()
            y = torch.from_numpy((batch.label - 1).double().data.numpy()).double() # batch.label is 1/2, while we want 0/1
            
            
            for i in range(x.shape[0]):
                log_likelhood_class0 = np.log(class_label_dir_posterior_expectation.numpy()[0])
                log_likelhood_class1 = np.log(class_label_dir_posterior_expectation.numpy()[1])
                
                np.place(x[i], x[i] > 10, 10)
                
                log_likelhood_class0 = np.sum(log_dir0_posterior_expectation[x[i].astype(int), np.arange(vocab_length)]) + np.log(class_label_dir_posterior_expectation.numpy()[0])
                log_likelhood_class1 = np.sum(log_dir1_posterior_expectation[x[i].astype(int), np.arange(vocab_length)]) + np.log(class_label_dir_posterior_expectation.numpy()[1])
                
                y_pred = float(np.array(np.argmax(np.array([log_likelhood_class0, log_likelhood_class1]))))
                
                test_num_correct.add_(y_pred == y[i])
                size_test_data.add_(1.0)
                                   
            return test_num_correct/size_test_data		
		
		
prior_param_magnitudes = [0.01, 0.05, 0.1, 0.4, 0.7, 1.0]
maxD = 10
val_accuracies_beta = torch.zeros(len(prior_param_magnitudes)).double()
val_accuracies_dir = torch.zeros(len(prior_param_magnitudes)).double()
max_val_accuracy_beta = torch.zeros(1).double()
max_val_accuracy_dir = torch.zeros(1).double

for i in range(len(prior_param_magnitudes)):
    
    beta0_prior_params = torch.DoubleTensor([prior_param_magnitudes[i], prior_param_magnitudes[i]])
    beta1_prior_params = torch.DoubleTensor([prior_param_magnitudes[i], prior_param_magnitudes[i]])
    class_labels_dir_prior_params = torch.DoubleTensor([1, 1])
    class_label_dir_posterior_expectation, beta0_posterior_expectation, beta1_posterior_expectation = NB_train(train_iter, class_labels_dir_prior_params, V, num_labels, \
                                                                                                              beta0_prior_params, beta1_prior_params)

    dir0_prior_params = torch.ones(V).double()*prior_param_magnitudes[i]
    dir1_prior_params = torch.ones(V).double()*prior_param_magnitudes[i]
    class_labels_dir_prior_params = torch.DoubleTensor([1, 1])
    class_label_dir_posterior_expectation, dir0_posterior_expectation, dir1_posterior_expectation = NB_train(train_iter, class_labels_dir_prior_params, V, num_labels, \
                                                                                                    dir0_unif_prior_param=prior_param_magnitudes[i], dir1_unif_prior_param=prior_param_magnitudes[i], \
                                                                                                    maxD=maxD, betaPrior=False)
    
    val_accuracies_beta[i] = compute_accuracy(val_iter, V, class_label_dir_posterior_expectation, beta0_posterior_expectation, beta1_posterior_expectation)[0]
    
    val_accuracies_dir[i] = compute_accuracy(val_iter, V, class_label_dir_posterior_expectation, dir0_posterior_expectation=dir0_posterior_expectation, \
                                             dir1_posterior_expectation=dir1_posterior_expectation, maxD=maxD, betaPrior=False)[0]
    
    if i == 0:
        max_val_accuracy_beta = val_accuracies_beta[0]
        max_val_accuracy_dir = val_accuracies_dir[0]
        best_beta_prior_param_magnitudes = prior_param_magnitudes[0]
        best_dir_prior_param_magnitudes = prior_param_magnitudes[0]
    
    if i > 0:
        if max_val_accuracy_beta < val_accuracies_beta[i]:
            max_val_accuracy_beta = val_accuracies_beta[i]
            best_beta_prior_param_magnitudes = prior_param_magnitudes[i]
            test_accuracy_best_beta_prior = compute_accuracy(test_iter, V, class_label_dir_posterior_expectation, beta0_posterior_expectation, beta1_posterior_expectation)[0]
        
        if max_val_accuracy_dir < val_accuracies_dir[i]:
            max_val_accuracy_dir = val_accuracies_dir[i]
            best_dir_prior_param_magnitudes = prior_param_magnitudes[i]
            test_accuracy_best_dir_prior = compute_accuracy(test_iter, V, class_label_dir_posterior_expectation, dir0_posterior_expectation=dir0_posterior_expectation, \
                                                            dir1_posterior_expectation=dir1_posterior_expectation, maxD=maxD, betaPrior=False)[0]
    print()        
    print('Val accuracy w/ beta class-conditional prior using uniform hyperparams of magnitude ' + str(prior_param_magnitudes[i]) + ': ' + str(val_accuracies_beta[i]))
    print('Val accuracy w/ Dirichlet class-conditional prior using uniform hyperparams of magnitude ' + str(prior_param_magnitudes[i]) + ': ' + str(val_accuracies_dir[i]))
    print()

print('Test accuracy w/ beta class-conditional prior using best uniform hyperparams (magnitude=' + \
      str(best_beta_prior_param_magnitudes) + '): ' + str(test_accuracy_best_beta_prior))
print('Test accuracy w/ Dirichlet class-conditional prior using best uniform hyperparams (magnitude=' + \
      str(best_dir_prior_param_magnitudes) + '): ' + str(test_accuracy_best_dir_prior))	
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		