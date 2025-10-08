import argparse

import numpy as np
import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
    # Load the diabetes dataset.
    dataset = sklearn.datasets.load_diabetes()

    # The input data are in `dataset.data`, targets are in `dataset.target`.

    t = dataset.target  #target vector(notation from lecture)

    
    # If you want to learn about the dataset, you can print some information
    # about it using `print(dataset.DESCR)`.


    # TODO: Append a constant feature with value 1 to the end of all input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.

    
    padding = np.ones((dataset.data.shape[0], 1))    #our vector of ones, which we'll use to pad in dataset.data
    X = np.append(dataset.data, padding, axis=1)    #our train set matrix

    
    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.

    
    X_train, X_test, t_train, t_test = sklearn.model_selection.train_test_split(X, t, test_size=args.test_size, random_state=args.seed)

    # TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).

    
    matrix_inverse = np.linalg.inv(X_train.T @ X_train)  #explicitly computing the matrix inverse(in lecture denoted as (X^T*X)^(-1))

    weights = (matrix_inverse @ X_train.T) @ t_train    

    # TODO: Predict target values on the test set.


    y_pred = X_test @ weights   #prediction of target values which is give by y(x)=x^T*w, where bias is hidden in w

    
    # TODO: Manually compute root mean square error on the test set predictions.

    err = y_pred -t_test #error vector, or distance between prediction and target value

    
    rmse = np.sqrt(np.mean((err)**2)) #Square the vector element-wise, and then I take the mean via np.mean() method,
    #and then of course we have to take the square root

    return rmse


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(main_args)
    print("{:.2f}".format(rmse))