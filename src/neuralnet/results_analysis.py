import numpy as np
from performance_func import plot_error


def extract_performance(models):
    for model in models:
        folds = model[0]['info']['folds']
        epochs = model[0]['params']['epochs']
        average_training_error = np.zeros(epochs)
        average_validation_error = np.zeros(epochs)
        average_validation_accuracy = np.zeros(epochs)
        overall_test_accuracy = 0
        for fold in range(folds-1): # Dit moet terug folds worden als alle 5 folds weer worden opgeslagen
            fold_model = model[fold]
            plot_error(fold_model['results']['errors'],fold_model['results']['val_errors'])
            average_training_error += fold_model['results']['errors']
            average_validation_error += fold_model['results']['val_errors']
            average_validation_accuracy += fold_model['results']['val_accs']
            overall_test_accuracy += fold_model['accuracies']
        average_training_error /= folds
        average_validation_error /= folds
        average_validation_accuracy /= folds
        overall_test_accuracy /= folds

        print(model[0]["info"]["description"]
        +" - layers " + str(model[0]["info"]["netparams"]["hidden_layers"])
        +" - epochs " + str(epochs)
        +" - learning_rate " + str(model[0]["params"]["learning_rate"]))
        print("Average_training_error: {} /n Average validation error: {} /n Average validation accuracy: {} /n"
              "Average test_accuracy: {}".format(average_training_error, average_validation_error,
                                                 average_validation_accuracy, overall_test_accuracy))
        plot_error(average_training_error,average_validation_error)


filename = "testing architectures - layers [256] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-04-175924.npy"
x = np.load("Results/TimsGroteBenchmark/"+filename)
extract_performance([x])



