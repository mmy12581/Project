## Summary of Common Machine Learning Techniques

## Naive Bayes Methods
\\[
P(A \cap B) = P(A) \cdot P(B|A) = P(B) \cdot P(B|A)
\\]
so that 
\\[
P(A | B) = \frac{P(B|A)\cdot P(A)}{P(B)}
\\]

###Process:
 
Assume that $$X = [a_1, a_2, ... a_n]$$, and $$X_s$$ are independent. There are K classes for $$Y$$. 

Our goal is to classify $$y_1,y_2,...,y_k$$, and essentially we are obtaining 
probabilities such that $$max(P(y_1 | X)),max(P(y_2 | X)),...max(P(y_k | X))$$. 

Known that

$$P(y_i | X) = \frac{P(X|y_i) \cdot P(y_i)}{P(X)}$$ 

for the classification goal, X is the same, so maximizing above equation is essentially maximizing 

$$P(X|y_i) \cdot P(y_i) = P(y_i) \cdot \prod_{i=1}^n P(a_i | y_i)$$

where $$P(a_i | y_i)$$ is the probability of $$a_i$$ shows up given class $$y_i$$ presented, and $$P(y_i)$$ is the probability of class $$y_i$$ presented over all samples.

###Notes:
1. When features are discrete, apply statistics (distrbution probability).
2. When features are continuous, assume features are normally distributed

###Problems:
1. When some of the features are not contributing to the classification, $$P(a | y) = 0$$, the performance of classification is getting worse. Thus, we can apply **Laplace Smoothing**, which gives the feature a small non-zero probability, so that the posteior probability do not suddenly drops to zero.

2. When features are not independent, we can use **DAG** describe the probability plot 

### Pros and Cons
Pros:

1. Computationally Simple and easy to train
2. Obtain high accuracy with small data
3. Not sensitive when there is missing data 

Cons:

1. Low accuracy when data is strongly correlated 
2. Sensitive to the explanotory data type


## Logistic Regression

$$
h_w(x) = \frac{1}{1 -(w^Tx + b))}
$$

Assume data is $$(x, y)$$, and $$y_i = 1$$ with probability $$p_i$$, and $$y_i = 0$$ with probability $$1 - p_i$$, so the expect probilty is 

$$
P(y_i) = h_w(x)^{y_i} \cdot (1 - h_w(x))^{1 - y_i}
$$

so that the likelihood is 

$$
P(w|y_i) = \prod_{i=1}^n h_w(x)^{y_i} \cdot (1 - h_w(x))^{1 - y_i}
$$

and the log-likelihood is 

$$
\ell(w) = \sum_{i=1}^n (y_i \cdot \log h_w(x_i) + (1 - y_i) \cdot \log(1-h_w(x_i))) = \sum_{i=1}^n (y_i \cdot (w^T x_i) + log(1 + e^{w^T x_i})) 
$$

Thus, we just need to find the MLE $$\hat{w}$$ maximizes $$\ell(w)$$. 

### Gradient Descent:
Loss function of $$\ell(w)$$ is 

$$
J(w) = -\frac{1}{N} \sum_{i=1}^n (y_i \cdot \log h_w(x_i) + (1 - y_i) \cdot \log(1-h_w(x_i)))
$$

so we are actually minimizing $$J(w)$$, and the process updating $$w$$ is 

$$
w := w - \alpha \cdot \triangledown J(w) 
$$

$$
w := w - \alpha \cdot \frac{1}{N} \cdot \sum_{i=1}^N(h_w(x_i) - y_i) x_i)
$$

where $$\alpha$$ is the step size, and the descent stops until $$J(w)$$ is minized.

The problem of gradient descent is that it leads to local minima, and for cost, it calculates over all samples, so it is slow. The way to solve it is to use the stochastic gradient descent 

$$
w := w - \alpha \cdot \frac{1}{N} \cdot (h_w(x_i) - y_i) x_i
$$

and SGD can use a dynamic step size 

$$
\alpha = 0.04 \cdot (1 + n + i) + r
$$

### Other Optimization Method (Fast but complex):
1. Newton's ralphson (use Hessian and cholesky decomposition)
2. BFGS or L-BFGS-B

### Overfitting Problem:
1. Reduce the numbers of features 
2. Regularzition $$(L_1, L_2)$$, so new loss function will be $$J(w) + \lambda||w||^2$$, and $$w:= w - \alpha \cdot 	(h_w(x_j) - y_j) x_j) - 2 \alpha w_j$$

### Multi-class Logistic Regression: softmax

$$
P(Y = a | x) = \frac{\exp(w_a \cdot x)}{\sum_{i=1}^n w_i \cdot x}; \quad 1 < a < k
$$

where $$\sum_{a=1}^n P(Y = a | X) = 1$$

### About choosing softmax or multiple logistic regression 
1. If classes are disjoint e.g images are human, dog, or cat, using softmax.
2. If classes are realted e.g a song might contain human's voice, dancing, and so on, we can use mulitple logistic regression

### Pros and Cons
Pros:

1. Easy to train
2. Computationally simple 

Cons:

1. Underfitting when the distribution of data is complex, so less accuracy
2. Only linearly seperable, so it is limited 


## K Nearest Neighbourhood 
$$
P(Y = j | X = x_0) = \frac{1}{K} \sum_{i \in N_0} I(y_i = j)
$$

where, $$N_0$$ is the neighbourhood that K points in the training dataset are closest to $$x_0$$.

### Three factors 
1. Choice of K
2. Distance Metrics, e.g euclidean, mahalobis
3. Rules of decision, e.g majority votes 

### Choice of K 
1. When k is small, the model is more complex, so overfitting
2. When k is large, model is more simple, and it will produce a decision boundary that is close to linear 

### Pros and Cons
Pros: 

1. Easy to achieve, and theory behind is strong 
2. It does not need assumption, so it is suitable for non-linear case
3. Training time is O(N)
4. High accuracy, and not sensitive to outliers 

Cons:

1. Large search problem to find nearest neighbours 
2. Storage of data
3. Hard to deal with unbalanced data 

### Notes
To have a quick search, KD trees can be used, and its time is O(log(N))


## Support Vector Mahcine
For sample data $$(x_i, y_i)$$, and the hyperplane $$w^Tx_i + b = 0$$

1. functional seperation: $$y_i(w^Tx_i + b)$$
2. geomery seperation: $$\frac{y_i(w^Tx_i + b)}{||w||}$$

### Linear SVM
\\[
\underset{w, b}{\mathrm{argmax}} \quad \gamma \\
st. \quad \frac{y_i(w^T x_i + b}{||w||} \geq \gamma 
\\]

Assume that $$\hat{\gamma} = \gamma \cdot ||w||$$, and then above becomes 

\\[
\underset{w, b}{\mathrm{argmax}} \quad \frac{\hat\gamma}{||w||} \\
st. \quad \frac{y_i(w^T x_i + b)}{||w||} \geq 1 
\\]

since the increase or decrease by times on $$\hat\gamma$$ does not affect the distance, so here $$\hat\gamma = 1$$, and also $$max(\frac{1}{||w||}) = min(\frac{1}{2} ||w||^2)$$, so above turns to 

\\[
\underset{w, b}{\mathrm{argmin}} \quad \frac{1}{2} ||w||^2 \\
st. \quad y_i(w^T x_i + b) \geq 1 
\\]

Thus, this is a convex function to solve, and we can apply Lagrange and then duality theory to solve it. Thus the final decsion boundary becomes 

\\[
f(x) = sign(\sum_{i=1}^N \alpha_i \cdot y_i(x_i \cdot x_i') + b)
\\]

where $$\alpha$$ is non-zero only for the support vectors in the solution
**which means that the decision boundary is dependent on the input $$x$$ and the input inner product.**

**Extension**:
Above is maximizing the hard boundary for SVM, and we use a slacky variable $$\xi$$ to maxmize the soft boundary for SVM such that 

\\[
\underset{w, b}{\mathrm{argmin}} \quad \frac{1}{2} \cdot ||w||^2 + C \sum_{i=1}^N \xi_i  \\
st. \quad y_i(w^T x_i + b) \geq 1 - \xi_i; \\
\xi_i \geq 0, i = 1,2,..n
\\]

**Loss Funciton :**
$$
Loss = \sum_{i=1}^N [1 - y_i(w^T x_i + b)]_{+} + \lambda ||w||^2
$$

### Pros and Cons
Pros:

1. Using kernel can project from low dimensional to high dimensional space
2. Radial Kernel can be really beneficial to non-linear classificaiton
3. Optimiality guaranteed due to the convex case
4. Comformity with semi-supervised learning since we only need to add additional condition to minimization problem, which is called **Transductive SVM**
5. High accuracy

Cons:

1. Require lots of memory and CPU time
2. Need to select a good kernel function
3. Not direct to multi class problems

### Sequetial Minimal Optimization
SMO breaks this problem into a series of smallest possible sub-problems, which are then solved analytically. Because of the linear equality constraint involving the Lagrange multipliers $$\alpha_i$$, the smallest possible problem involves two such multipliers. Then, for any two multipliers $$\alpha_1  \alpha_2$$, the constraints are reduced to:
\\[
0 \geq \alpha_1, \alpha_2 \geq C\\
y_1\alpha_1 + y_2\alpha_2 = k
\\]

and this reduced problem can be solved analytically: one needs to find a minimum of a one-dimensional quadratic function. {\displaystyle k} k is the negative of the sum over the rest of terms in the equality constraint, which is fixed in each iteration.

### Multi-class SVM
1. one to many 
2. one to one
