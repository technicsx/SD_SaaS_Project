from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def output_model_confusion_matrix(model, y_test, y_pred):
    # Calculate the confusion matrix
    confusion_matrix_res = confusion_matrix(y_test, y_pred)
    print(confusion_matrix_res)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_res,
                                  display_labels=model.classes_)
    disp.plot()
