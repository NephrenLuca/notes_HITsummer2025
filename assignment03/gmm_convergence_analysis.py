import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import cv2
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')

class GMMConvergenceAnalysis:
    def __init__(self, image_path):
        """Initialize the GMM convergence analysis with an image path."""
        self.image_path = image_path
        self.image = None
        self.rg_chromaticities = None
        self.best_gmm = None
        self.worst_gmm = None
        self.convergence_results = []
        
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
    
    def run_multiple_initializations(self, data, n_components=3, n_init=100):
        """
        Run multiple GMM initializations and collect all results.
        """
        print(f"Running {n_init} GMM initializations with {n_components} components...")
        
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
            
            # Store results
            self.convergence_results.append({
                'initialization': i,
                'gmm': gmm,
                'score': score,
                'means': gmm.means_,
                'weights': gmm.weights_,
                'covariances': gmm.covariances_
            })
            
            print(f"  Initialization {i+1}: Log likelihood = {score:.4f}")
        
        # Sort results by score (best to worst)
        self.convergence_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Store best and worst results
        self.best_gmm = self.convergence_results[0]['gmm']
        self.worst_gmm = self.convergence_results[-1]['gmm']
        
        print(f"\nBest convergence: Initialization {self.convergence_results[0]['initialization']+1}")
        print(f"  Log likelihood: {self.convergence_results[0]['score']:.4f}")
        print(f"Worst convergence: Initialization {self.convergence_results[-1]['initialization']+1}")
        print(f"  Log likelihood: {self.convergence_results[-1]['score']:.4f}")
        
        return self.convergence_results
    
    def plot_gmm_comparison(self, data, best_gmm, worst_gmm, title="Best vs Worst GMM Convergence"):
        """
        Plot comparison between best and worst GMM fits.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # High contrast colors for contours
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # Plot best convergence
        ax1.scatter(data[:, 0], data[:, 1], alpha=0.3, s=0.5, c='lightgray', label='Data points')
        
        for i in range(best_gmm.n_components):
            mean = best_gmm.means_[i]
            cov = best_gmm.covariances_[i]
            
            # Create grid for contour plotting
            x_min, x_max = data[:, 0].min() - 0.1, data[:, 0].max() + 0.1
            y_min, y_max = data[:, 1].min() - 0.1, data[:, 1].max() + 0.1
            
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            pos = np.dstack((xx, yy))
            
            # Calculate probability density
            rv = multivariate_normal(mean, cov)
            z = rv.pdf(pos)
            
            # Find 3σ contour
            peak_prob = rv.pdf(mean)
            contour_level = peak_prob * np.exp(-9/2)
            
            # Plot contour
            color = colors[i % len(colors)]
            ax1.contour(xx, yy, z, levels=[contour_level], colors=color, linewidths=3)
            ax1.scatter(mean[0], mean[1], c=color, marker='o', s=100, edgecolors='black', linewidth=2)
        
        ax1.set_xlabel('r chromaticity', fontsize=12)
        ax1.set_ylabel('g chromaticity', fontsize=12)
        ax1.set_title('Best Convergence', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.2)
        ax1.set_facecolor('white')
        
        # Plot worst convergence
        ax2.scatter(data[:, 0], data[:, 1], alpha=0.3, s=0.5, c='lightgray', label='Data points')
        
        for i in range(worst_gmm.n_components):
            mean = worst_gmm.means_[i]
            cov = worst_gmm.covariances_[i]
            
            # Create grid for contour plotting
            x_min, x_max = data[:, 0].min() - 0.1, data[:, 0].max() + 0.1
            y_min, y_max = data[:, 1].min() - 0.1, data[:, 1].max() + 0.1
            
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            pos = np.dstack((xx, yy))
            
            # Calculate probability density
            rv = multivariate_normal(mean, cov)
            z = rv.pdf(pos)
            
            # Find 3σ contour
            peak_prob = rv.pdf(mean)
            contour_level = peak_prob * np.exp(-9/2)
            
            # Plot contour
            color = colors[i % len(colors)]
            ax2.contour(xx, yy, z, levels=[contour_level], colors=color, linewidths=3)
            ax2.scatter(mean[0], mean[1], c=color, marker='o', s=100, edgecolors='black', linewidth=2)
        
        ax2.set_xlabel('r chromaticity', fontsize=12)
        ax2.set_ylabel('g chromaticity', fontsize=12)
        ax2.set_title('Worst Convergence', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.2)
        ax2.set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig('gmm_best_vs_worst_convergence.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def plot_convergence_statistics(self):
        """
        Plot statistics of all convergence results.
        """
        if not self.convergence_results:
            print("No convergence results available. Run run_multiple_initializations first.")
            return
        
        scores = [result['score'] for result in self.convergence_results]
        init_numbers = [result['initialization'] + 1 for result in self.convergence_results]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Score distribution
        ax1.hist(scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.4f}')
        ax1.axvline(np.median(scores), color='green', linestyle='--', label=f'Median: {np.median(scores):.4f}')
        ax1.set_xlabel('Log Likelihood Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Convergence Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Score vs Initialization
        ax2.scatter(init_numbers, scores, alpha=0.6, color='orange')
        ax2.set_xlabel('Initialization Number')
        ax2.set_ylabel('Log Likelihood Score')
        ax2.set_title('Score vs Initialization')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Box plot of scores
        ax3.boxplot(scores, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
        ax3.set_ylabel('Log Likelihood Score')
        ax3.set_title('Box Plot of Convergence Scores')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative distribution
        sorted_scores = np.sort(scores)
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        ax4.plot(sorted_scores, cumulative, 'b-', linewidth=2)
        ax4.set_xlabel('Log Likelihood Score')
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cumulative Distribution of Scores')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gmm_convergence_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_detailed_comparison(self):
        """
        Print detailed comparison between best and worst convergence.
        """
        if not self.best_gmm or not self.worst_gmm:
            print("No GMM results available. Run run_multiple_initializations first.")
            return
        
        print("\n" + "="*60)
        print("DETAILED COMPARISON: BEST vs WORST CONVERGENCE")
        print("="*60)
        
        best_result = self.convergence_results[0]
        worst_result = self.convergence_results[-1]
        
        print(f"\nBEST CONVERGENCE (Initialization {best_result['initialization']+1}):")
        print(f"  Log Likelihood: {best_result['score']:.4f}")
        print("  Component Parameters:")
        for i in range(3):
            mean = best_result['means'][i]
            weight = best_result['weights'][i]
            print(f"    Component {i+1}: Mean=({mean[0]:.3f}, {mean[1]:.3f}), Weight={weight:.3f}")
        
        print(f"\nWORST CONVERGENCE (Initialization {worst_result['initialization']+1}):")
        print(f"  Log Likelihood: {worst_result['score']:.4f}")
        print("  Component Parameters:")
        for i in range(3):
            mean = worst_result['means'][i]
            weight = worst_result['weights'][i]
            print(f"    Component {i+1}: Mean=({mean[0]:.3f}, {mean[1]:.3f}), Weight={weight:.3f}")
        
        print(f"\nSTATISTICS:")
        scores = [result['score'] for result in self.convergence_results]
        print(f"  Score Range: {min(scores):.4f} to {max(scores):.4f}")
        print(f"  Score Difference: {max(scores) - min(scores):.4f}")
        print(f"  Standard Deviation: {np.std(scores):.4f}")
        print(f"  Coefficient of Variation: {np.std(scores)/np.mean(scores):.4f}")
    
    def run_analysis(self, n_init=100):
        """
        Run complete convergence analysis.
        """
        print(f"\n{'='*60}")
        print(f"GMM CONVERGENCE ANALYSIS (K=3)")
        print(f"{'='*60}")
        
        # Load image
        image = self.load_image()
        
        # Convert to rg chromaticity space
        r, g = self.rgb_to_rg_chromaticity(image)
        self.rg_chromaticities = (r, g)
        
        # Prepare data for GMM
        rg_data_valid, valid_mask = self.prepare_data_for_gmm(r, g)
        
        # Run multiple initializations
        convergence_results = self.run_multiple_initializations(rg_data_valid, n_components=3, n_init=n_init)
        
        # Plot comparison
        self.plot_gmm_comparison(rg_data_valid, self.best_gmm, self.worst_gmm)
        
        # Plot statistics
        self.plot_convergence_statistics()
        
        # Print detailed comparison
        self.print_detailed_comparison()
        
        return convergence_results

def main():
    """Main function to run the convergence analysis."""
    # Initialize the analysis object
    analyzer = GMMConvergenceAnalysis('GMMSegmentTestImage.jpg')
    
    # Run analysis with 20 initializations
    results = analyzer.run_analysis(n_init=100)
    
    print("\nAnalysis complete! Check the generated PNG files for visualizations.")
    print("- gmm_best_vs_worst_convergence.png: Side-by-side comparison")
    print("- gmm_convergence_statistics.png: Statistical analysis of all runs")

if __name__ == "__main__":
    main()