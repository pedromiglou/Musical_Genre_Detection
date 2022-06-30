import matplotlib.pyplot as plt

def plot_accuracy_comparison(accs, title, legend):
    epochs = len(accs[0])
    plt.figure(figsize = (10,5))
    for acc in accs:
        plt.plot(range(1, epochs+1), acc)

    plt.xticks(range(1, epochs+1))
    plt.title(title)
    plt.legend(legend)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()


def plot_loss_comparison(losses, title, legend):
    epochs = len(losses[0])
    plt.figure(figsize = (10,5))
    for loss in losses:
        plt.plot(range(1, epochs+1), loss)

    plt.xticks(range(1, epochs+1))
    plt.title(title)
    plt.legend(legend)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


def plot_confusion_matrix(y_test, y_pred):
    plt.matshow(confusion_matrix(y_test, y_pred))
    plt.ylabel("Predicted Category", fontsize=14)
    plt.title("Category", fontsize=14)
    plt.show()