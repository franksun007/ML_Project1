# Annotation for [ML Advice] (http://cs229.stanford.edu/materials/ML-advice.pdf)

1. Diagnostics for debugging learning algorithms.
2. Error analyses and ablative analysis.
3. How to get started on a machine learning problem.
    - Premature (statistical) optimization.

# Traditionally the correct approach (which is wrong)
        Common approach: Try improving the algorithm in different ways.
            – Try getting more training examples.
            – Try a smaller set of features.
            – Try a larger set of features.
            – Try changing the features: Email header vs. email body features.
            – Run gradient descent for more iterations.
            – Try Newton’s method.
            – Use a different value for λ.
            – Try using an SVM.

        This approach might work, but it’s very time-consuming, and largely a matter
        of luck whether you end up fixing what the problem really is.

# Better Approach 
1. Better approach:
    - Run diagnostics to figure out what the problem is.
    - Fix whatever the problem is.
2. Suppose you suspect the problem is either:
    - Overfitting (high variance).
    - Too few features to classify spam (high bias).
3. Diagnostic:
    - Variance: Training error will be much lower than test error.
    - Bias: Training error will also be high.

# Apprach summary
1. Approach #1: Careful design.
    - Spend a long term designing exactly the right features, collecting the right dataset, and designing the right algorithmic architecture.
    - Implement it and hope it works.
    - Benefit: Nicer, perhaps more scalable algorithms. May come up with new, elegant, learning algorithms; contribute to basic research in machine learning.
2. Approach #2: Build-and-fix.
    - Implement something quick-and-dirty.
    - Run error analyses and diagnostics to see what’s wrong with it, and fix its errors.
    - Benefit: Will often get your application problem working more quickly. Faster time to market.
