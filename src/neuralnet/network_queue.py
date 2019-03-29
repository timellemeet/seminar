from data_func import k_fold
from network import Network

class Queue:
    def __init__(self, x, y):
        self.queue = []
        self.features = x
        self.labels = y

    def add(self,netparams, folds, params):
        for i in range(folds):
            data = k_fold(self.features, self.labels, k=folds, i=i+1)
            self.queue.append({
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
                ),
                "data":data,
                "params":params})

    def execute(self):
        print(self.queue)
        results = [None] * len(self.queue)
        accuracies = [None] * len(self.queue)
        for i, val in enumerate(self.queue):
            print("Fitting model %d/%d" %(i+1,len(self.queue)))
            results[i] = val["network"].fit(
                            val["data"]["x_train"],
                             val["data"]["y_train"],
                             val["data"]["x_val"],
                             val["data"]["y_val"],
                             val["params"]["epochs"],
                             val["params"]["learning_rate"],
                             val["params"]["batch_size"],
                             val["params"]["momentum"],
                             val["params"]["weight_decay"])
            accuracies[i] = val["network"].

        return results
