# Annotation for [Useful things to know about machine learning] (http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)

## LEARNING = REPRESENTATION + EVALUATION + OPTIMIZATION

The most common mistake among machine learning beginners is to test on the training data and have the illusion of success.


Bias and Variance: 
This can be mitigated by doing cross-validation: randomly dividing your training data into (say) ten subsets, holding out each one while training on the rest, testing each learned classifier on the examples it did not see, and averaging the results to see how well the particular parameter setting does.

Cross-validation can help to combat overfitting, for example by using it to choose the best size of decision tree to learn.

Besides cross-validation, there are many methods to combat overfitting. The most popular one is adding a regularization term to the evaluation function.

## Understand Overfitting by decomposing Generalization Error into Bias and Variance

Bias is a learner’s tendency to consistently learn the same wrong thing.
Variance is the tendency to learn random things irrespective of the real signal.


This [slide] (https://courses.cs.washington.edu/courses/cse546/12wi/slides/cse546wi12LinearRegression.pdf)  is pretty useful


## LEARN MANY MODELS, NOT JUST ONE
Because we never know which one will be better for the current existing data we have, and that kind of data in the future. 

## CORRELATION DOES NOT IMPLY CAUSATION
Which is always true... 
