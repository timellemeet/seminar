from results_analysis import Model
from performance_func import plot_confusion_matrix, heatmatrix
import numpy as np

# Results = [
#
# ]
#
# model1 = Model("Results/Crossplot/Baseresults - layers [10, 10] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-05-183742.npy")
# model2 = Model("Results/Crossplot/Baseresults - layers [20, 10] - training_size 60000 - epochs 20 - learning_rate 0.005 - 2019-04-05-231446.npy")
#
#
# model2.summary()
# model2.plot_error()


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

# for key, value in Results.items():
#         value.overall_test_accuracy = round(value.overall_test_accuracy,3)

tablevals = np.empty([5, 5], dtype=object)
# for key, value in Results.items():
#         print(key, round(value.overall_test_accuracy,3)

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

print(tablevals)




labels = ["50","40","30","20","10"]
print(heatmatrix(tablevals, labels, labels))