'''
Created on 09 gen 2018

@author: Umberto
'''

from random import seed

from ml.MultilayerNnClassifier import MultilayerNnClassifier
from ml.activation.Sigmoid import Sigmoid
from DataPreparation import DataPreparation
from evaluation.ClassificationEvaluator import ClassificationEvaluator
from evaluation.Splitting import Splitting
import matplotlib.pyplot as plt
def classificationSeed():
    '''
    Test Classification on Seeds dataset
    '''
        
    seed(1)
    
    n_folds = 5
    l_rate = 0.2
    n_epoch = 50
    n_hidden = [10,10]
    
    mlp = MultilayerNnClassifier();
    activationFunction = Sigmoid()
    dp = DataPreparation();
    evaluator = ClassificationEvaluator();
    splitting = Splitting();
   
    # load and prepare data
    filename = '../Datasets/data_banknote_authentication.txt'
    dataset = dp.load_csv(filename)
    for i in range(len(dataset[0]) - 1):
        dp.str_column_to_float(dataset, i)
    # convert class column to integers
    dp.str_column_to_int(dataset, len(dataset[0]) - 1)
    # normalize input variables
    minmax = dp.dataset_minmax(dataset)
    dp.normalize_dataset_classification(dataset, minmax)    
    # evaluate algorithm
    scores,errors = evaluator.evaluate_algorithm(dataset, splitting, mlp.back_propagation, n_folds, l_rate, n_epoch, n_hidden, activationFunction)  
    print_classification_scores(scores,errors) 


def print_classification_scores(scores,errors):
    print('Scores: %s' % scores)
    print('errors: %s' % errors)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))   
    print('Mean errors: %.3f%%' % (sum(errors) / float(len(errors))))  
    plt.plot([1,2,3,4,5],errors) 
    plt.axis([0, 5, 0, 5])
    plt.ylabel("Errors %")
    plt.xlabel("Epochs")
    plt.show()

        
if __name__ == '__main__':
    classificationSeed()