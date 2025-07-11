import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
# log_likelihood_score doesn't exist in sklearn.metrics
# We'll use the score() method from the GMM object instead
import cv2
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')

class GMMImageSegmentation:
    def __init__(self, image_path):
        """Initialize the GMM segmentation with an image path."""
        self.image_path = image_path
        self.image = None
        self.rg_chromaticities = None
        self.gmm = None
        self.posterior_probs = None
        
    def load_image(self):
        """Load the image and convert to RGB format."""
        # Load image using OpenCV (BGR format)
        self.image = cv2.imread(self.image_path)
        # Convert BGR to RGB
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        print(f"Image loaded with shape: {self.image.shape}")
        return self.image
    
    def rgb_to_rg_chromaticity(self, image):
        """
        Convert RGB image to rg chromaticity space.
        
        r = R / (R + G + B)
        g = G / (R + G + B)
        """
        # Normalize RGB values to [0, 1]
        image_norm = image.astype(np.float64) / 255.0
        
        # Calculate sum of RGB channels
        rgb_sum = np.sum(image_norm, axis=2)
        
        # Avoid division by zero
        rgb_sum = np.where(rgb_sum == 0, 1, rgb_sum)
        
        # Calculate rg chromaticities
        r = image_norm[:, :, 0] / rgb_sum
        g = image_norm[:, :, 1] / rgb_sum
        
        return r, g
    
    def rg_to_rgb_visualization(self, r, g):
        """
        Convert rg chromaticities back to RGB for visualization.
        
        α = 255 / max(r, g, 1-(r+g))
        Rout = round(αr)
        Gout = round(αg)
        Bout = round(α(1-(r+g)))
        """
        # Calculate b = 1 - (r + g)
        b = 1 - (r + g)
        
        # Calculate α
        alpha = 255 / np.maximum(np.maximum(r, g), b)
        
        # Convert back to RGB
        R_out = np.round(alpha * r).astype(np.uint8)
        G_out = np.round(alpha * g).astype(np.uint8)
        B_out = np.round(alpha * b).astype(np.uint8)
        
        # Stack channels
        rgb_out = np.stack([R_out, G_out, B_out], axis=2)
        
        return rgb_out
    
    def prepare_data_for_gmm(self, r, g):
        """Prepare rg chromaticities for GMM fitting."""
        # Reshape to 2D array where each row is a pixel's rg values
        height, width = r.shape
        rg_data = np.column_stack([r.flatten(), g.flatten()])
        
        # Remove invalid pixels (where r + g > 1 or negative values)
        valid_mask = (r.flatten() + g.flatten() <= 1) & (r.flatten() >= 0) & (g.flatten() >= 0)
        rg_data_valid = rg_data[valid_mask]
        
        print(f"Valid pixels for GMM: {rg_data_valid.shape[0]} out of {rg_data.shape[0]}")
        
        return rg_data_valid, valid_mask
    
    def fit_gmm_multiple_initializations(self, data, n_components, n_init=10):
        """
        Fit GMM with multiple random initializations and select the best one.
        """
        best_gmm = None
        best_score = -np.inf
        
        print(f"Fitting GMM with {n_components} components using {n_init} initializations...")
        
        for i in range(n_init):
            # Initialize GMM with random parameters
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                random_state=i,
                max_iter=1000,
                tol=1e-4
            )
            
            # Fit the model
            gmm.fit(data)
            
            # Calculate log likelihood
            score = gmm.score(data)
            
            if score > best_score:
                best_score = score
                best_gmm = gmm
                print(f"  Initialization {i+1}: Log likelihood = {score:.4f} (new best)")
            else:
                print(f"  Initialization {i+1}: Log likelihood = {score:.4f}")
        
        self.gmm = best_gmm
        print(f"Best GMM selected with log likelihood: {best_score:.4f}")
        return best_gmm
    
    def plot_gmm_contours(self, data, gmm, title="GMM Fit Visualization"):
        """
        Plot scatter plot of rg chromaticities with GMM contours.
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use a more distinguishable color scheme for data points
        # Use a light gray for data points to avoid conflict with contour colors
        ax.scatter(data[:, 0], data[:, 1], alpha=0.3, s=0.5, c='lightgray', label='Data points')
        
        # Plot means with high contrast colors
        means = gmm.means_
        ax.scatter(means[:, 0], means[:, 1], c='black', marker='x', s=300, linewidths=4, label='Component means')
        
        # Use high contrast colors for contours that are easily distinguishable
        # Avoid blue since data points were blue, and use colors that stand out
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        
        for i in range(gmm.n_components):
            mean = gmm.means_[i]
            cov = gmm.covariances_[i]
            
            # Create grid for contour plotting
            x_min, x_max = data[:, 0].min() - 0.1, data[:, 0].max() + 0.1
            y_min, y_max = data[:, 1].min() - 0.1, data[:, 1].max() + 0.1
            
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            pos = np.dstack((xx, yy))
            
            # Calculate probability density
            rv = multivariate_normal(mean, cov)
            z = rv.pdf(pos)
            
            # Find 3σ contour (probability = exp(-9/2) ≈ 0.0111 of peak)
            peak_prob = rv.pdf(mean)
            contour_level = peak_prob * np.exp(-9/2)
            
            # Plot contour with thicker lines and better colors
            color = colors[i % len(colors)]
            ax.contour(xx, yy, z, levels=[contour_level], colors=color, 
                      linewidths=3, label=f'Component {i+1}')
            
            # Also plot the mean point with the same color for better association
            ax.scatter(mean[0], mean[1], c=color, marker='o', s=100, edgecolors='black', linewidth=2)
        
        ax.set_xlabel('r chromaticity', fontsize=12)
        ax.set_ylabel('g chromaticity', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.2)
        
        # Set background to white for better contrast
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig(f'gmm_contours_k{gmm.n_components}.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def compute_posterior_probabilities(self, r, g, valid_mask):
        """
        Compute posterior probabilities for each pixel belonging to each component.
        """
        # Prepare data
        rg_data = np.column_stack([r.flatten(), g.flatten()])
        rg_data_valid = rg_data[valid_mask]
        
        # Compute posterior probabilities
        posterior_probs = self.gmm.predict_proba(rg_data_valid)
        
        # Initialize full posterior probability array
        height, width = r.shape
        full_posterior = np.zeros((height * width, self.gmm.n_components))
        full_posterior[valid_mask] = posterior_probs
        
        # Reshape to image dimensions
        self.posterior_probs = full_posterior.reshape(height, width, self.gmm.n_components)
        
        return self.posterior_probs
    
    def visualize_posterior_probabilities(self, posterior_probs):
        """
        Visualize posterior probabilities as grayscale images.
        """
        n_components = posterior_probs.shape[2]
        fig, axes = plt.subplots(1, n_components, figsize=(4*n_components, 4))
        
        if n_components == 1:
            axes = [axes]
        
        for i in range(n_components):
            # Extract probability map for component i
            prob_map = posterior_probs[:, :, i]
            
            # Convert to 8-bit grayscale (0-255)
            prob_map_8bit = (prob_map * 255).astype(np.uint8)
            
            # Display
            axes[i].imshow(prob_map_8bit, cmap='gray')
            axes[i].set_title(f'Component {i+1} Posterior Probability')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'posterior_probabilities_k{n_components}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_posterior_probability_images(self, posterior_probs, base_filename):
        """
        Save posterior probabilities as individual grayscale images.
        """
        n_components = posterior_probs.shape[2]
        
        for i in range(n_components):
            # Extract probability map for component i
            prob_map = posterior_probs[:, :, i]
            
            # Convert to 8-bit grayscale (0-255)
            prob_map_8bit = (prob_map * 255).astype(np.uint8)
            
            # Save image
            filename = f"{base_filename}_component_{i+1}.png"
            cv2.imwrite(filename, prob_map_8bit)
            print(f"Saved {filename}")
    
    def run_analysis(self, n_components=3):
        """
        Run complete GMM analysis for given number of components.
        """
        print(f"\n{'='*50}")
        print(f"GMM Analysis with {n_components} components")
        print(f"{'='*50}")
        
        # Load image
        image = self.load_image()
        
        # Convert to rg chromaticity space
        r, g = self.rgb_to_rg_chromaticity(image)
        self.rg_chromaticities = (r, g)
        
        # Visualize rg chromaticity image
        rg_rgb = self.rg_to_rgb_visualization(r, g)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(image)
        axes[0].set_title('Original RGB Image')
        axes[0].axis('off')
        
        axes[1].imshow(rg_rgb)
        axes[1].set_title('RG Chromaticity Visualization')
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig('rgb_vs_rg_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Prepare data for GMM
        rg_data_valid, valid_mask = self.prepare_data_for_gmm(r, g)
        
        # Fit GMM with multiple initializations
        gmm = self.fit_gmm_multiple_initializations(rg_data_valid, n_components)
        
        # Visualize GMM fit
        self.plot_gmm_contours(rg_data_valid, gmm, f"GMM Fit with {n_components} Components")
        
        # Compute and visualize posterior probabilities
        posterior_probs = self.compute_posterior_probabilities(r, g, valid_mask)
        self.visualize_posterior_probabilities(posterior_probs)
        self.save_posterior_probability_images(posterior_probs, f"posterior_k{n_components}")
        
        return gmm, posterior_probs

def main():
    """Main function to run the complete analysis."""
    # Copy the test image to this directory
    import shutil
    import os
    
    # Initialize the segmentation object
    seg = GMMImageSegmentation('GMMSegmentTestImage.jpg')
    
    # Run analysis for K = 3, 4, 5
    results = {}
    
    for k in [3, 4, 5]:
        print(f"\n{'='*60}")
        print(f"ANALYSIS FOR K = {k} COMPONENTS")
        print(f"{'='*60}")
        
        gmm, posterior_probs = seg.run_analysis(n_components=k)
        results[k] = {'gmm': gmm, 'posterior_probs': posterior_probs}
        
        # Print component information
        print(f"\nGMM Component Information (K={k}):")
        for i in range(k):
            mean = gmm.means_[i]
            weight = gmm.weights_[i]
            print(f"  Component {i+1}: Mean = ({mean[0]:.3f}, {mean[1]:.3f}), Weight = {weight:.3f}")
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON OF DIFFERENT K VALUES")
    print(f"{'='*60}")
    
    for k in [3, 4, 5]:
        gmm = results[k]['gmm']
        # Load image again for comparison
        image = cv2.imread('GMMSegmentTestImage.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r, g = seg.rgb_to_rg_chromaticity(image)
        rg_data_valid, valid_mask = seg.prepare_data_for_gmm(r, g)
        score = gmm.score(rg_data_valid)
        print(f"K={k}: Log likelihood = {score:.4f}")
    
    print("\nComments on different K values:")
    print("- K=3: May capture the main color regions in the image")
    print("- K=4: Could provide more detailed segmentation")
    print("- K=5: May over-segment and capture noise")
    
    # Create summary plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, k in enumerate([3, 4, 5]):
        gmm = results[k]['gmm']
        rg_data_valid, _ = seg.prepare_data_for_gmm(seg.rg_chromaticities[0], seg.rg_chromaticities[1])
        
        # Plot scatter with means
        axes[i].scatter(rg_data_valid[:, 0], rg_data_valid[:, 1], alpha=0.3, s=1, c='blue')
        means = gmm.means_
        axes[i].scatter(means[:, 0], means[:, 1], c='red', marker='+', s=200, linewidths=3)
        axes[i].set_title(f'K={k} Components')
        axes[i].set_xlabel('r chromaticity')
        axes[i].set_ylabel('g chromaticity')
        axes[i].grid(True, alpha=0.3)
    
    # Plot log likelihood comparison
    k_values = [3, 4, 5]
    scores = []
    for k in k_values:
        gmm = results[k]['gmm']
        rg_data_valid, _ = seg.prepare_data_for_gmm(seg.rg_chromaticities[0], seg.rg_chromaticities[1])
        score = gmm.score(rg_data_valid)
        scores.append(score)
    
    axes[3].bar(k_values, scores, color=['blue', 'green', 'red'])
    axes[3].set_title('Log Likelihood Comparison')
    axes[3].set_xlabel('Number of Components (K)')
    axes[3].set_ylabel('Log Likelihood')
    axes[3].grid(True, alpha=0.3)
    
    # Plot component weights
    for i, k in enumerate([4, 5]):
        gmm = results[k]['gmm']
        weights = gmm.weights_
        component_labels = [f'C{i+1}' for i in range(len(weights))]
        axes[4+i].pie(weights, labels=component_labels, autopct='%1.1f%%')
        axes[4+i].set_title(f'Component Weights (K={k})')
    
    plt.tight_layout()
    plt.savefig('gmm_analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nAnalysis complete! Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main() 