from pathvalidate import sanitize_filename
from data_func import k_fold
from network import Network
import numpy as np
import os
import time


class Queue:
    def __init__(self, x, y, x_test, y_test, y_true):
        self.queue = []
        self.info = []
        self.features = x
        self.labels = y
        self.test_features = x_test
        self.test_labels = y_test
        self.original_test_labels = y_true

    def add(self,description,netparams, folds, params):
        self.info.append({"description":description, "folds":folds, "netparams":netparams , "params":params })
        for i in range(folds):
            data = k_fold(self.features, self.labels, k=folds, i=i+1)
            self.queue.append({
                "description": description + " - Fold "+str(i+1),
                "network":Network(
                    netparams["hidden_layers"],
                    netparams["features"],
                    netparams["output_classes"],
                    netparams["activation"],
                    netparams["activation_prime"],
                    netparams["loss_activation"],
                    netparams["loss_activation_prime"],
                    netparams["loss"],
                    netparams["loss_prime"],
                    netparams["activation_alpha"],
                ),
                "data":data,
                "params":params,
                "results":None,
                "accuracies":None})

    def execute(self, save=True, folder="Results"):
        infoindex = 0
        startindex = 0
        endindex = self.info[infoindex]["folds"] - 1
        
        for i, val in enumerate(self.queue):
            print("Fitting model %d/%d" %(i+1,len(self.queue)))
            self.queue[i]["results"] = val["network"].fit(
                            val["data"]["x_train"],
                             val["data"]["y_train"],
                             val["data"]["x_val"],
                             val["data"]["y_val"],
                             val["params"]["epochs"],
                             val["params"]["learning_rate"],
                             val["params"]["batch_size"],
                             val["params"]["momentum"],
                             val["params"]["weight_decay"])
            self.queue[i]["accuracies"]= val["network"].accuracy(self.test_features, self.original_test_labels)
            print("Model accuracy ",self.queue[i]["accuracies"])
            del self.queue[i]["data"]
            
            
            #save folded batch
            if save and endindex == i:
                timestamp = time.strftime("%Y-%m-%d-%H%M%S")
                os.makedirs(folder, exist_ok=True)
                np.save(folder+"/"+sanitize_filename(
                    self.info[infoindex]["description"]
                    +" - layers "+str(self.info[infoindex]["netparams"]["hidden_layers"])
                    +" - training_size "+str(self.labels.shape[0])
                    +" - epochs "+str(val["params"]["epochs"])
                    +" - learning_rate "+str(val["params"]["learning_rate"])
                    +" - "+timestamp), 
                        self.queue[startindex:endindex])
                
                if i < len(self.queue) - 1:
                    startindex = endindex + 1
                    infoindex += 1
                    endindex += self.info[infoindex]["folds"]
        
        return self.queue
    