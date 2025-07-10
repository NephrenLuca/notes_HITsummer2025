### Assignment 01

**A posteriori probability**

Let $P(X|v=u)$ denote the probability of drawing out $X$ blacks in a total of $N=10$ draws from urn $v$. Obviously, the probability of drawing out a black each time from urn $v$ is 
$$
p_v\equiv P(color=black|v=u)=\frac{u}{10}
$$
Considering the process of drawing 10 balls to be a Bernoulli distribution, we have the probability
$$
\begin{align}
P(X=3|v=u)&=C_{10}^{3}\times p_v^3\times (1-p_v)^7\\
&=120\times (\frac{u}{10})^3 \times (1-\frac{u}{10})^7
\end{align}
$$
Applying Bayesian law we have
$$
P(v=u|X=3)=\frac{P(X=3|v=u)\times P(v=u)}{P(X=3)}
$$
where $P(v=u)=\frac1{11}$, and $P(X=3)$ can be calculated as
$$
P(X=3)=\frac1{11}\sum_{i=0}^{10}P(X=3|v=i)
$$
Applying the figures we have results as follows:

| $u$  | $P(v=u|X=3)$ |
| ---- | ------------ |
| 0    | 0            |
| 1    | 0.063073     |
| 2    | 0.221240     |
| 3    | **0.293220** |
| 4    | 0.236256     |
| 5    | 0.128779     |
| 6    | 0.046668     |
| 7    | 0.009892     |
| 8    | 0.000864     |
| 9    | 0.000010     |
| 10   | 0            |

where maximum probability occurs when $u = 3$.

**Computer Generation of Random Variables**

*(a)*

Since $Y=G(X)$, we have $P(Y\le y)=P(G(X)\le y)$.

According to the question, $F(t)$ must be monotone increasing and differentiable. Applying $F(⋅)$ to both sides we have
$$
P(G(X)\le y)=P(F(G(X))\le F(y))=P(X\le F(y))
$$
Because $X∼Uniform(0,1)$, its CDF takes the form
$$
P(X\le x)=x=F(y),~~~~\forall u\in [0,1)
$$
Thus
$$
P(Y\le y)=F(y)
$$
Applying the derivative to both side we have
$$
p_Y(y)=p(y)
$$


*(b)*

For $N=10^4$, results are as follows:![image-20250708142202269](C:\Users\14144\AppData\Roaming\Typora\typora-user-images\image-20250708142202269.png)

![image-20250708142238275](C:\Users\14144\AppData\Roaming\Typora\typora-user-images\image-20250708142238275.png)

For $N=10^6$, we have:

![image-20250708142335459](C:\Users\14144\AppData\Roaming\Typora\typora-user-images\image-20250708142335459.png)

![image-20250708142403061](C:\Users\14144\AppData\Roaming\Typora\typora-user-images\image-20250708142403061.png)

