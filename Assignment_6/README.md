Shows neural network using Pytorch framework for the MNIST handwritten digits.

The code has 5 different types of models with differently tuned hyper-parameters:

1. L1 regularizer implemented with Batch-Normalization.
2. L2 regularizer implemented with Batch-Normalization.
3. L1+L2 regularizer both implemented with Batch-Normalization.
4. Ghost Batch-Normalization.
5. L1+L2 regularizer implemented with Ghost Batch-Normalization.

Training logs are visible showing the results for 25 epochs each.

Every model has its Training loss, Test loss, Training accuracy, Test accuracy Curves.

ONE graph is shown for the validation accuracy curves for all 5 jobs above.

ONE graph is shown for the loss change curves for all 5 jobs above.


25 misclassified images for every model is shown with Target predictions and Actual predictions.
