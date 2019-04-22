# Github repo for seminar A layered approach to classifying digits
This repo contains all code used for the seminar from team 11.
All code can be found in the source code directory, where each folder contains the code for different classifiers. For all classifiers,
the required packages were loaded in a jupyter notebook to be able run easily. When programming our own packages, pycharm was used to help 
coding. For k-nearest neighbours and the logistic regression
we used the sklearn package, and for the neural network only numpy was used for the matrix algebra. 
## neuralnet directory
The neuralnet directory is where most of our code resides. The main components used for the neural network are two python files,
Layer which contains all different layer classes, and network which is creates the network class. There are however more files
which contain functions used in both these main classes:
- activation_func file is used to import all the different activation functions a network can take.
- data_func contains functions to import the data, vectorize the data, and perform cross-validation
- loss_func is used to import the loss functions with their derivatives.
- performance_func contains two functions to analyse the performance of networks.

To combine these functions and classes, and to be able to easily run multiple networks after each other the Queue class in network_queue
was created. This is the main function which is called in every notebook to run all sorts of networks.
All the different notebooks in the neuralnet folder were used to run different configurations, to run different networks.
## Results
The Queue class can save a network with corresponding output, training and validation errors, and the configuration using the savez function
build into numpy. To load the networks, a Model class was built in the results_analysis file. For an example on how it was used see the
analysis notebook. All networks were saved in the Results folder to their corresponding folder.
## Data
The data was downloaded directly from the LeCun website, however this was not in a readily made format as it is compressed heavily. 
A python library was used to open and parse the data into a csv file. Loading csvs into numpy is quite slow, so instead we loaded it once,
and then saved it using the savez function from numpy. To load it, we built the import_data function located in the data_func file.
Sadly, the file was too large to push onto github, and instead we uploaded it onto google drive. Here is a link to download the .npz file,
place it in the src folder. https://drive.google.com/file/d/1YxNhsyUG9VIMUmjHJelIkU5mWOllsB-F/view?usp=drivesdk

