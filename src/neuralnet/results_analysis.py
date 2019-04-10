import numpy as np
from performance_func import plot_error
import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from adjustText import adjust_text
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

def extract_model(model):
    folds = model[0]['info']['folds']
    epochs = model[0]['params']['epochs']
    average_training_error = np.zeros(epochs)
    average_validation_error = np.zeros(epochs)
    average_validation_accuracy = np.zeros(epochs)
    overall_test_accuracy = 0
    for fold in range(folds - 1):  # Dit moet terug folds worden als alle 5 folds weer worden opgeslagen
        fold_model = model[fold]
     #   plot_error(fold_model['results']['errors'], fold_model['results']['val_errors'])
        average_training_error += fold_model['results']['errors']
        average_validation_error += fold_model['results']['val_errors']
        average_validation_accuracy += fold_model['results']['val_accs']
        overall_test_accuracy += fold_model['accuracies']
    average_training_error /= folds - 1  # Dit moet terug folds worden als alle 5 folds weer worden opgeslagen
    average_validation_error /= folds - 1  # Dit moet terug folds worden als alle 5 folds weer worden opgeslagen
    average_validation_accuracy /= folds - 1  # Dit moet terug folds worden als alle 5 folds weer worden opgeslagen
    overall_test_accuracy /= folds - 1  # Dit moet terug folds worden als alle 5 folds weer worden opgeslagen

    print(model[0]["info"]["description"]
          + " - layers " + str(model[0]["info"]["netparams"]["hidden_layers"])
          + " - epochs " + str(epochs)
          + " - learning_rate " + str(model[0]["params"]["learning_rate"]))
    print("Average_training_error: {} \n Average validation error: {} \n Average validation accuracy: {} \n"
          "Average test_accuracy: {}".format(average_training_error, average_validation_error,
                                             average_validation_accuracy, overall_test_accuracy))
    # plot_error(average_training_error, average_validation_error)
    return [average_training_error, average_validation_error, average_validation_accuracy, overall_test_accuracy]


def extract_performance(models):
    all_results = []
    for model in models:
        all_results.append(extract_model(model))

    plot_error(all_results[0][2], all_results[1][2],
               "Architecture: {}".format(models[0][0]["info"]["netparams"]["hidden_layers"]),
               "Architecture: {}".format(models[1][0]["info"]["netparams"]["hidden_layers"]),
               x_axis='epochs', y_axis='validation accuracy')

def operations_plot(models):
    texts = []
    # mpl.use('pgf')
    # params = {
    #     'font.family': 'serif',
    #     'text.usetex': True,
    #     'text.latex.unicode': True,
    #     'pgf.rcfonts': False,
    #     'pgf.texsystem': 'xelatex'
    # }
    # mpl.rcParams.update(params)
    with open('../plots/more than 2 layers.txt', 'w') as writer1:
        writer1.write(str(('ops', 'acc', 'layer structure')) + '\n')
        with open('../plots/2 layers.txt', 'w') as writer2:
            writer2.write(str(('ops', 'acc', 'layer structure')) + '\n')
            for model in models:
                acc = model.overall_test_accuracy
                layers = model.model[0]['info']['netparams']['hidden_layers']
                operations = 784*layers[0]+10*layers[-1]
                for hu in layers[1:]:
                    operations += hu*hu
                fig = plt.gcf()
                fig.set_size_inches(15,10)

                if len(layers) == 2:
                    plt.scatter(operations, acc, color='b')
                    texts.append(plt.text(operations, acc, str(layers), ha='center', va='bottom'))
                    writer2.write(str((operations, acc, layers)) + '\n')
                if len(layers) > 2:
                    plt.scatter(operations, acc, color='y')
                    texts.append(plt.text(operations, acc, str(layers), ha='center', va='bottom'))
                    writer1.write(str((operations, acc, layers)) + '\n')

    plt.show()


def load_models():
    #returns a list of Model objects
    models = []
    for file in os.listdir('../neuralnet/Results/Base/'):
        if file[-4:] != '.npy':
            continue
        models.append(Model('../neuralnet/Results/Base/'+file))
    return models
models = load_models()
operations_plot(models)
# filename = "testing architectures - layers [100] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-05-054805.npy"
# x = np.load("Results/TimsGroteBenchmark/"+filename)
# filename2 = "testing architectures - layers [100, 90] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-05-000259.npy"
# y = np.load("Results/TimsGroteBenchmark/"+filename2)
# extract_performance([x, y])
