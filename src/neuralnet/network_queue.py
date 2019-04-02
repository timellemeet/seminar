from data_func import k_fold
from network import Network

class Queue:
    def __init__(self, x, y, x_test, y_test, y_true):
        self.queue = []
        self.features = x
        self.labels = y
        self.test_features = x_test
        self.test_labels = y_test
        self.original_test_labels = y_true

    def add(self,description,netparams, folds, params):
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

    def execute(self):
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
            del self.queue[i]["data"]
        return self.queue
