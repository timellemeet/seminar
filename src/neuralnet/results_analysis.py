import numpy as np
from performance_func import plot_error
from data_func import import_data
import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from adjustText import adjust_text
import datetime
import matplotlib.patches as mpatches
import matplotlib2tikz

class Model:
    def __init__(self, path):
        self.model = np.load(path)
        folds = self.model[0]['info']['folds']
        epochs = self.model[0]['params']['epochs']
        self.description = (self.model[0]["info"]["description"]
            + " - layers " + str(self.model[0]["info"]["netparams"]["hidden_layers"])
            + " - epochs " + str(epochs)
            + " - learning_rate " + str(self.model[0]["params"]["learning_rate"]))
        self.average_training_error = np.zeros(epochs)
        self.average_validation_error = np.zeros(epochs)
        self.average_validation_accuracy = np.zeros(epochs)
        self.overall_test_accuracy = 0
        self.overall_apt = 0 #average epoch time
        for fold in range(folds):
            fold_model = self.model[fold]
         #   plot_error(fold_model['results']['errors'], fold_model['results']['val_errors'])
            self.average_training_error += fold_model['results']['errors']
            self.average_validation_error += fold_model['results']['val_errors']
            self.average_validation_accuracy += fold_model['results']['val_accs']
            self.overall_test_accuracy += fold_model['accuracies']
            self.overall_apt += fold_model['results']['apt']
        self.average_training_error /= folds
        self.average_validation_error /= folds
        self.average_validation_accuracy /= folds
        self.overall_test_accuracy /= folds
        self.overall_apt /= folds
        self.overall_apt = str(datetime.timedelta(seconds=round(self.overall_apt)))

    def summary(self):
        print(self.description)
        print("Average_training_error: {} \n Average validation error: {} \n Average validation accuracy: {} \n"
              "Average test_accuracy: {}".format(self.average_training_error, self.average_validation_error,
                                                 self.average_validation_accuracy, self.overall_test_accuracy))
        print("Overall average epoch time: ",self.overall_apt)
        print('\n')
        #plot_error(average_training_error, average_validation_error)
        #return [average_training_error, average_validation_error, average_validation_accuracy, overall_test_accuracy]

    def plot_error(self, save=''):
            plot_error(self.average_training_error, self.average_validation_error, save)
            
    def toplosses(self, amount=10, fold=1):
        np.random.seed(10)
        # import data
        normalize = True
        training, labels, test, original_test_labels, test_labels = import_data(size=60000, normalize=normalize)
        self.model[fold-1]["network"].top_losses(test, test_labels, amount)

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

    print()
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
    with open('../plots/2 layers.txt', 'w') as writer1:
        writer1.write(str(('ops', 'acc', 'layer structure')) + '\n')
        with open('../plots/1 layers.txt', 'w') as writer2:
            writer2.write(str(('ops', 'acc', 'layer structure')) + '\n')
            for model in models:
                acc = model.overall_test_accuracy
                layers = model.model[0]['info']['netparams']['hidden_layers']
                operations = 784*layers[0]+10*layers[-1] + layers[0] + 10
                for i in range(1, len(layers)):
                    #weights + bias
                    operations += layers[i-1]*layers[i]+layers[i]
                print("%s has %g operations" %(layers, operations))
                fig = plt.gcf()
                fig.set_size_inches(15,10)

                if len(layers) == 1:
                    plt.scatter(operations, acc, color='b')
                    texts.append(plt.text(operations, acc, str(layers), ha='center', va='bottom'))
                    # writer2.write(str((operations, acc, layers)) + '\n')
                if len(layers) == 2:
                    plt.scatter(operations, acc, color='y')
                    texts.append(plt.text(operations, acc, str(layers), ha='center', va='bottom'))
                    writer1.write(str((operations, acc, layers)) + '\n')

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.savefig('../plots/1 layer and 2 layers.png')
    plt.show()

def load_models(path):
    #returns a list of Model objects
    models = []
    for file in os.listdir(path):
        if file[-4:] != '.npy':
            continue
        models.append(Model(path+"/"+file))
    return models
def list_models(models):
    for i, val in enumerate(models):
        print("Key "+str(i)+": "+val.description)
    
# models = load_models()
# operations_plot(models)
models = load_models('../neuralnet/Results/base/')
# operations_plot(models)
for model in models:
    texts = []
    acc = model.overall_test_accuracy
    layers = model.model[0]['info']['netparams']['hidden_layers']
    operations = 784*layers[0]+10*layers[-1] + layers[0] + 10
    for i in range(1, len(layers)):
        #weights + bias
        operations += layers[i-1]*layers[i]+layers[i]
    if len(layers) <= 2 and layers[0]<=200:
        plt.plot(operations, acc,marker='o',markersize=4, color='b')
#         texts.append(plt.text(operations, acc, str(layers), ha='center', va='bottom'))

models_wd = load_models('../neuralnet/Results/Base/WeightDecay/')
for model in models_wd:
    texts = []
    acc = model.overall_test_accuracy
    layers = model.model[0]['info']['netparams']['hidden_layers']
    operations = 784*layers[0]+10*layers[-1] + layers[0] + 10
    for i in range(1, len(layers)):
        #weights + bias
        operations += layers[i-1]*layers[i]+layers[i]
    fig = plt.gcf()
    fig.set_size_inches(15,10)

    if model.model[0]['info']['params']['weight_decay'] == 0.0005:
        plt.plot(operations, acc,marker='o',markersize=4, color='y')
#         texts.append(plt.text(operations, acc, str(layers), ha='center', va='bottom'))
    elif model.model[0]['info']['params']['weight_decay'] == 0.001:
        plt.plot(operations, acc, marker='o',markersize=4,color='g')
#         texts.append(plt.text(operations, acc, str(layers), ha='center', va='bottom'))
    elif model.model[0]['info']['params']['weight_decay'] == 0.005:
        plt.plot(operations, acc, marker='o',markersize=4,color='m')
#         texts.append(plt.text(operations, acc, str(layers), ha='center', va='bottom'))
    elif model.model[0]['info']['params']['weight_decay'] == 0.01:
        plt.plot(operations, acc, marker='o',markersize=4,color='r')
#         texts.append(plt.text(operations, acc, str(layers), ha='center', va='bottom'))
red_patch = mpatches.Patch(color='r', label='wd = 0.01')
green_patch = mpatches.Patch(color='g', label='wd = 0.001')
mag_patch = mpatches.Patch(color='m', label='wd = 0.005')
yellow_patch = mpatches.Patch(color='y', label='wd = 0.0005')
blue_patch = mpatches.Patch(color='b', label='wd = 0')
patches = [red_patch, mag_patch, green_patch,yellow_patch, blue_patch]
legend = plt.gca().legend(handles=patches,loc='lower right')

# plt.legend(['lr=0.0001','lr=0.0005','lr=0.001','lr=0.005','lr=0.01'])
matplotlib2tikz.save('../plots/wd.tex')
# filename = "testing architectures - layers [100] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-05-054805.npy"
# x = np.load("Results/TimsGroteBenchmark/"+filename)
# filename2 = "testing architectures - layers [100, 90] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-05-000259.npy"
# y = np.load("Results/TimsGroteBenchmark/"+filename2)
# extract_performance([x, y])
