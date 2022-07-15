import pandas as pd
import os
import joblib
import argparse

from sklearn import metrics
from sklearn import tree
import config
import model_dispatcher

def run(fold, model):
    df = pd.read_csv(config.TRAINING_FILE)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # Drop the label column and convert into numpy array
    x_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values

    x_valid = df_valid.drop("label", axis=1).values
    y_valid = df_valid.label.values

    clf = model_dispatcher.models[model] #Initialize the classifier
    clf.fit(x_train, y_train) #fit the model on training data

    preds = clf.predict(x_valid) #create predictions for the validation dataset

    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT,f"dt_{fold}.bin" ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser() # initialize Argument Parser of argparse class
    parser.add_argument("--fold",type=int)
    parser.add_argument("--model", type=str)

    args = parser.parse_args()  # read the arguments from the command line
    run(fold=args.fold, model=args.model) # run the folds specified by command line arguments


    # run(fold=0)
    # run(fold=1)
    # run(fold=2)
    # run(fold=3)
    # run(fold=4)


