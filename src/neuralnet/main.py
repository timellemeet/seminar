Results = {
    "5010": Model("Results/Crossplot/confusion - layers [50, 10] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-08-123145.npy"),
    "5020": Model("Results/Crossplot/confusion - layers [50, 20] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-08-125515.npy"),
    "5030": Model("Results/Crossplot/confusion - layers [50, 30] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-08-123219.npy"),
    "5040": Model("Results/Crossplot/Baseresults - layers [50, 40] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-06-002804.npy"),
    "5050": Model("Results/Crossplot/Baseresults - layers [50, 50] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-05-195904.npy"),
    "4010": Model("Results/Crossplot/confusion - layers [40, 10] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-08-125215.npy"),
    "4020": Model("Results/Crossplot/Baseresults - layers [40, 20] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-06-031706.npy"),
    "4030": Model("Results/Crossplot/Baseresults - layers [40, 30] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-06-000232.npy"),
    "4040": Model("Results/Crossplot/Baseresults - layers [40, 40] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-05-193445.npy"),
    "3010": Model("Results/Crossplot/confusion - layers [30, 10] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-08-130737.npy"),
    "3020": Model("Results/Crossplot/Baseresults - layers [30, 20] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-05-233706.npy"),
    "3030": Model("Results/Crossplot/Baseresults - layers [30, 30] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-05-191320.npy"),
    "2010": Model("Results/Crossplot/Baseresults - layers [20, 10] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-05-231446.npy"),
    "2020": Model("Results/Crossplot/Baseresults - layers [20, 20] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-05-185412.npy"),
    "1010": Model("Results/Crossplot/Baseresults - layers [10, 10] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-05-183742.npy"),
}

tablevals = np.empty([5, 5], dtype=object)

#50s
tablevals[0,0] = round(Results["5050"].overall_test_accuracy,3)
tablevals[1,0] = round(Results["5040"].overall_test_accuracy,3)
tablevals[2,0] = round(Results["5030"].overall_test_accuracy,3)
tablevals[3,0] = round(Results["5020"].overall_test_accuracy,3)
tablevals[4,0] = round(Results["5010"].overall_test_accuracy,3)

#40s
tablevals[1,1] = round(Results["4040"].overall_test_accuracy,3)
tablevals[2,1] = round(Results["4030"].overall_test_accuracy,3)
tablevals[3,1] = round(Results["4020"].overall_test_accuracy,3)
tablevals[4,1] = round(Results["4010"].overall_test_accuracy,3)

#30s
tablevals[2,2] = round(Results["3030"].overall_test_accuracy,3)
tablevals[3,2] = round(Results["3020"].overall_test_accuracy,3)
tablevals[4,2] = round(Results["3010"].overall_test_accuracy,3)

#20s
tablevals[3,3] = round(Results["2020"].overall_test_accuracy,3)
tablevals[4,3] = round(Results["2010"].overall_test_accuracy,3)

#10
tablevals[4,4] = round(Results["1010"].overall_test_accuracy,3)

labels = ["50","40","30","20","10"]
heatmatrix(tablevals, labels, labels)
# =======
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# # from sklearn.metrics import accuracy_score
# from Layer import *
# from network import Network
# from activation_func import tanh, tanh_prime, sigmoid, sigmoid_prime, softmax, softmax_prime, \
#     reloid, reloid_prime, relu, relu_prime, leaky_relu, leaky_relu_prime
# from loss_func import mse, mse_prime, cross_entropy, cross_entropy_prime
# from data_func import vectorize_labels, k_fold, import_data
# from performance_func import plot_error, plot_confusion_matrix
# from network_queue import Queue
# np.random.seed(10)
# # import data
# training_size = 3000
# normalize = True
# training, labels, test, original_test_labels, test_labels = import_data(size=training_size, normalize=normalize)

# # intializing queue
# queue = Queue(training, labels, test, test_labels, original_test_labels)

# # specify input and output sizes
# features = 784
# output_classes = 10

# # hyper parameters
# learning_rate = 5e-3
# hidden_layers = [30]
# max_epochs = 5
# batch_size = 1
# weight_decay = 0.01
# momentum = True
# reloid_alpha = 0.5
# leaky_relu_alpha = 0.01

# architectures = [[30]]
# for layers in architectures:
#     queue.add(netparams={
#         "hidden_layers": layers,
#         "features": features,
#         "output_classes": output_classes,
#         "activation": leaky_relu,
#         "activation_prime": leaky_relu_prime,
#         "activation_alpha": leaky_relu_alpha,
#         "loss_activation": softmax,
#         "loss_activation_prime": softmax_prime,
#         "loss": cross_entropy,
#         "loss_prime": cross_entropy_prime
#         },
#         folds=5,
#         params={"epochs": 5,
#                 "learning_rate": 5e-3,
#                 "batch_size": 1,
#                 "momentum": False,
#                 "weight_decay": 0.},
#         description="architecture: "+str(layers)+" training_size: "+str(training_size))

# results_queue = queue.execute(save=False)

# # # prepare data for training by selecting validation set
# # fold_train_data, fold_train_labels, fold_val_data, fold_val_labels = k_fold(training, labels, k=5, n=5)
# #
# # # train the model on training data and labels using specific hyper-parameters
# # errors, val_errors, val_accs = net.fit(fold_train_data, fold_train_labels, fold_val_data, fold_val_labels,
# #                                        max_epochs, learning_rate, batch_size, momentum, weight_decay)

# # train the model on training data and labels using specific hyper-parameters
# # errors, val_errors, val_accs = net.fit(fold_train_data, fold_train_labels, fold_val_data, fold_val_labels,
# #                              max_epochs, learning_rate, batch_size, momentum, weight_decay)
# # net.save_parameters("%s HU, %f LR, %d epochs, %d batchsize, %f weightdecay, %g momentum" %(hidden_layers, learning_rate, max_epochs, batch_size, weight_decay, momentum))
# # # print the accuracy
# # print("The test accuracy of the network is: {}".format(
# #     net.accuracy(x=test, y_true=original_test_labels, errors=errors, val_errors=val_errors)))

# # # plot and print performance measures
# # plot_confusion_matrix(y_pred, original_test_labels[:test_size], classes=np.array([0,1,2,3,4,5,6,7,8,9]),
# #                       normalize=True,
# #                       title='Normalized confusion matrix')
# # plt.show()
# # plot_error(accuracies[:5], accuracies[5:10])
# >>>>>>> 4361f5119976bd1681fac596165a0800258c916b
