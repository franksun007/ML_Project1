# README for run.py

## Execution
```python3 run.py```

## Dependencies
1. ```numpy``` lib
2. ```proj1_helpers.py```

```proj1_helpers.py``` should be under the same folder as run.py

## Input
There are two input files for the program, one is the training dataset, 
another is the test dataset

1. training.csv
2. test.csv

Both files can be specified in run.py under variable:
```
DATA_TRAIN_PATH = '..'   # Training  data path
DATA_TEST_PATH = '..'    # Test      data path
```

## Output
Besides stdout, there is an output files which indicates our prediction

1. output.csv 

The file can be specified in run.py under variable:
```
OUTPUT_PATH = '...'     # Output file path
```

## Feature Engineering
1. Splitting Data

    We split the training data into three sets according to PRI_jet_num
    
    1. PRI_jet_num is 0
    2. PRI_jet_num is 1
    3. PRI_jet_num is 2 or 3
    
2. Standardizing Data

    After the splitting of the data, we get rid of the useless features. 
    Specifically we kept 18 features when PRI_jet_num is 0, 22 when PRI_jet_num is 1 
    and 29 when PRI_jet_num is 2 or 3 (get rid of the PRI_jet_num feature)
    
    After the above operations, we normalize the data to mean 0, standard deviation 1.
    The mean is calculated without the outliers (-999), and the outliers are replaced
    by the mean. The standard deviation is performed after the normalizing mean and outliers.
    
3. Additional Feature

    We tried to add **sin** and **tanh** features to the dataset, but experiments
    shown they did not provide better classification results. **No** additional features
    were added.
    
4. Polynomial Expansion

    We increased function to 2-order polynomial. We tried to do 3, 4 and even 5 but the 
    experiments shown that 2-order polynomial expansion converges the best and gave the 
    best result. 
    
    
## Model 
We used a regularized logistic regression with gradient descent to perform the 
classification, with the above feature engineering. 

We set a max iteration time to be 40000, and the threshold for loss 1e-8. 

We set a lambda size of 0.1 and a initial Gamma size of 0.005. 

A dynamic Gamma search was performed through the iteration. If the current loss value 
is larger than the previous loss value, then the Gamma is reduced to 2/3 of the original value. 


## Validation
We did not perform cross validation in terms of k-fold. 

A ```performance``` function was coded to compare the result of classification. 
However, we used the training data to determine how well we are doing. This function
has the potential to give very good result, but overfit the training dataset. 

Because of the model we chose and the feature engineering we performed, we still chose
to use this function in order to evaluate how we are doing. However, the significance of
this function is to indicate whether we did something very wrong or not, instead
of telling us exactly how well we are doing. 


# Result

This script achieved around 40 on the public leader board, with an accuracy of around 81.3%