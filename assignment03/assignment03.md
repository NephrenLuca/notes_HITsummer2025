### Assignment 03

**Image Segmentation Using GMMs**

*(a)*

Comparison between the image before and after transformation is presented as follows:

![rgb_vs_rg_comparison](.\rgb_vs_rg_comparison.png)

*Preserved attributes*

- Color hue: The fundamental color (red, green, blue, etc.) is preserved

- Relative color relationships: The ratios between R, G, B channels are maintained

- Chromaticity: The color quality is independent of brightness

- Color boundaries: Edges between different colored regions remain intact

*lost attributes*

- Brightness/intensity: All luminance information is discarded

- Absolute color values: The actual RGB values are lost

- Shadows and highlights: Dark and bright areas become indistinguishable

*(b)*

$K=3$![gmm_contours_k3](.\gmm_contours_k3.png)

![posterior_probabilities_k3](.\posterior_probabilities_k3.png)

$K=4$

![gmm_contours_k4](.\gmm_contours_k4.png)

![posterior_probabilities_k4](.\posterior_probabilities_k4.png)

$K=5$

![gmm_contours_k5](.\gmm_contours_k5.png)

![posterior_probabilities_k5](.\posterior_probabilities_k5.png)

*Comments*

![gmm_analysis_summary](.\gmm_analysis_summary.png)

Pros and cons for different values of $K$:

| $K$  | Pros                                                         | Cons                               |
| ---- | ------------------------------------------------------------ | ---------------------------------- |
| 3    | Captures the main color regions in the image；Simple, interpretable segmentation | May miss subtle color variations   |
| 4    | Provides more detailed segmentation；Better fit to data (higher likelihood) | More complex model                 |
| 5    | Most detailed segmentation；Highest likelihood fit           | May over-segment and capture noise |

In fact, component 2 of the case where $K=5$ has an unusual mean value and a low weight, indicating that it might be capturing noise.

Different random initializations produced varying results. Looking at results we have the following key observations:

- Most initializations converged to similar high-quality solutions
- A few initializations may converge to a poor local optimum
- Multiple initializations are crucial for robust GMM fitting

For simplicity's sake, we only give the demonstration of differences between good and bad GMM initializations where $K=3$. After 100 runs of random initialized GMMs, we compare the best and worst result as follows:

![gmm_best_vs_worst_convergence](.\gmm_best_vs_worst_convergence.png)

The difference is very slight, and all runs converge to local optimums very near each other. Local optimums might result in poor results in capturing the characteristics of images, but in the given image in question this disturbance is negligible.

---

*sklearn and scipy are used for the GMM realization in this report.*

