import numpy as np
from performance_func import plot_error
from data_func import import_data


class Model:
    def __init__(self, path):
        self.model = np.load(path)
        folds = self.model[0]['info']['folds']
        epochs = self.model[0]['params']['epochs']
        self.average_training_error = np.zeros(epochs)
        self.average_validation_error = np.zeros(epochs)
        self.average_validation_accuracy = np.zeros(epochs)
        self.overall_test_accuracy = 0
        for fold in range(folds):
            fold_model = self.model[fold]
         #   plot_error(fold_model['results']['errors'], fold_model['results']['val_errors'])
            self.average_training_error += fold_model['results']['errors']
            self.average_validation_error += fold_model['results']['val_errors']
            self.average_validation_accuracy += fold_model['results']['val_accs']
            self.overall_test_accuracy += fold_model['accuracies']
        self.average_training_error /= folds
        self.average_validation_error /= folds
        self.average_validation_accuracy /= folds
        self.overall_test_accuracy /= folds

    def summary(self):
        print(self.model[0]["info"]["description"]
              + " - layers " + str(self.model[0]["info"]["netparams"]["hidden_layers"])
              + " - epochs " + str(self.model[0]['params']['epochs'])
              + " - learning_rate " + str(self.model[0]["params"]["learning_rate"]))
        print("Average_training_error: {} \n Average validation error: {} \n Average validation accuracy: {} \n"
              "Average test_accuracy: {}".format(self.average_training_error, self.average_validation_error,
                                                 self.average_validation_accuracy, self.overall_test_accuracy))
        #plot_error(average_training_error, average_validation_error)
        #return [average_training_error, average_validation_error, average_validation_accuracy, overall_test_accuracy]

    def plot_error(self):
            plot_error(self.average_training_error, self.average_validation_error)
            
    def toplosses(self, amount=10, fold=1):
        np.random.seed(10)
        # import data
        normalize = True
        training, labels, test, original_test_labels, test_labels = import_data(size=60000, normalize=normalize)
        self.model[fold-1]["network"].top_losses(test, test_labels, amount)

