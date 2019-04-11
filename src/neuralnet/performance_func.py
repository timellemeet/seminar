import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


def plot_error(x1, x2, x1_name='training loss', x2_name='validation loss', x_axis='iteration', y_axis='loss'):
    fig, ax1 = plt.subplots()
    ax1.plot(x1, 'r', label="{} ({:.6f})".format(x1_name, x1[-1]))
    ax1.plot(x2, 'b--', label="{} ({:.6f})".format(x2_name, x2[-1]))
    ax1.grid(True)
    ax1.set_xlabel(x_axis)
    ax1.legend(loc="best", fontsize=9)
    ax1.set_ylabel(y_axis, color='r')
    ax1.tick_params('y', colors='r')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
#     classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

def heatmatrix(a, ylabels, xlabels, title="Table Title", background="black", font="white", margin=0.02):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    xlabel = "xlabel"
    ylabel = "ylabel"

    shape = a.shape

    if len(shape) > 2:
        raise ValueError('tabular can at most display two dimensions')

    columns = "c" * (shape[1] + 1)

    #minimize
    flat_list = []
    for sublist in a:
        for item in sublist:
            if item != None: flat_list.append(item)

    lb = min(flat_list) * (1 - margin)
    ub = max(flat_list) * (1 + margin)
    def cellvalue(x):
        if x == None: return ""
        else:
            cellcolor = r'\cellcolor{'+background+'!'+str(int(round(100*(x-lb)/(ub-lb))))+'} '
            cellvalue = r'\textcolor{'+font+'}{'+str(x)+'}'
            return cellcolor + cellvalue

    a = np.vectorize(cellvalue)(a)

    rv = [r'\begin{table}[h!]']
    rv += [r'\centering']
    rv += [r'\captionof{table}{'+title+'} ']
    rv += [r'\begin{tabular}{'+columns+'}'] #columns[:-1]
    rv += [" & " + ' & '.join(xlabels)+ r"\\ "]
    for i, line in enumerate(a):
        rv += [ylabels[i] + " & " +' & '.join(line)+ r"\\ "] #\hline

    #rv = rv[:-5]
    rv +=  [r'\end{tabular}']
    rv += [r'\end{table}']
    print('\n'.join(rv))

