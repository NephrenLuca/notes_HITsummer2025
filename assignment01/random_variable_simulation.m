% Random Variable Generation using Inverse Transform Method
% PDF: p(u) = 2u for 0 ≤ u ≤ 1, 0 otherwise

clear; clc; close all;

%% Parameters
N = 1e4;  % Number of realizations
u_max = 1;
u_min = 0;

%% Step 1: Find the CDF and its inverse
% PDF: p(u) = 2u for 0 ≤ u ≤ 1
% CDF: F(u) = ∫p(t)dt from 0 to u = ∫2t dt = t²|₀ᵘ = u²
% Inverse CDF: F⁻¹(y) = √y

%% Step 2: Generate uniform random variables
X = rand(N, 1);  % Uniform random variables in [0,1]

%% Step 3: Apply inverse transform to get desired distribution
Y = sqrt(X);  % Y = F⁻¹(X) = √X

%% Step 4: Create histogram and compare with theoretical PDF
% Create histogram
nbins = 50;
[counts, edges] = histcounts(Y, nbins);
bin_centers = (edges(1:end-1) + edges(2:end)) / 2;
bin_width = edges(2) - edges(1);

% Normalize histogram to match PDF
% The histogram counts need to be normalized so that the area equals 1
% Area = sum(counts) * bin_width
% To normalize: normalized_counts = counts / (sum(counts) * bin_width)
normalized_counts = counts / (sum(counts) * bin_width);

% Theoretical PDF values at bin centers
theoretical_pdf = 2 * bin_centers;  % p(u) = 2u

%% Step 5: Plotting
figure('Position', [100, 100, 800, 600]);

% Plot histogram and theoretical PDF
bar(bin_centers, normalized_counts, 1, 'FaceColor', [0.7, 0.7, 0.9], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.7);
hold on;
plot(bin_centers, theoretical_pdf, 'r-', 'LineWidth', 2);

% Customize plot
xlabel('u', 'FontSize', 12);
ylabel('Probability Density', 'FontSize', 12);
title('Random Variable Generation: p(u) = 2u', 'FontSize', 14);
legend('Normalized Histogram', 'Theoretical PDF', 'Location', 'northwest');
grid on;
xlim([0, 1]);

% Add text box with explanation
text(0.02, 1.8, sprintf(['Normalization Procedure:\n' ...
    '• Counts divided by (total counts × bin width)\n' ...
    '• Ensures histogram area = 1\n' ...
    '• Makes histogram comparable to PDF\n' ...
    'N = %d realizations'], N), ...
    'FontSize', 10, 'VerticalAlignment', 'top', ...
    'BackgroundColor', 'white', 'EdgeColor', 'black');

%% Step 6: Verification
% Check if histogram area is approximately 1
histogram_area = sum(normalized_counts) * bin_width;
fprintf('Histogram area: %.6f (should be close to 1)\n', histogram_area);

% Calculate theoretical area
theoretical_area = trapz(bin_centers, theoretical_pdf);
fprintf('Theoretical PDF area: %.6f\n', theoretical_area);

% Calculate mean and variance
sample_mean = mean(Y);
sample_var = var(Y);
theoretical_mean = 2/3;  % ∫u·p(u)du = ∫2u²du = 2/3
theoretical_var = 1/18;  % Var = E[u²] - (E[u])² = 1/2 - (2/3)² = 1/18

fprintf('\nSample Statistics:\n');
fprintf('Sample mean: %.6f (theoretical: %.6f)\n', sample_mean, theoretical_mean);
fprintf('Sample variance: %.6f (theoretical: %.6f)\n', sample_var, theoretical_var);

%% Step 7: Additional verification plot
figure('Position', [100, 700, 800, 400]);

% Plot CDF comparison
u_vals = linspace(0, 1, 1000);
theoretical_cdf = u_vals.^2;  % F(u) = u²

% Empirical CDF
[ecdf_vals, ecdf_x] = ecdf(Y);

subplot(1, 2, 1);
plot(u_vals, theoretical_cdf, 'r-', 'LineWidth', 2);
hold on;
plot(ecdf_x, ecdf_vals, 'b--', 'LineWidth', 1.5);
xlabel('u');
ylabel('Cumulative Probability');
title('CDF Comparison');
legend('Theoretical CDF', 'Empirical CDF', 'Location', 'southeast');
grid on;

% Plot QQ plot
subplot(1, 2, 2);
theoretical_quantiles = sqrt(linspace(0, 1, N+1));  % F⁻¹(p) = √p
theoretical_quantiles = theoretical_quantiles(2:end);  % Remove 0
sample_quantiles = sort(Y);
plot(theoretical_quantiles, sample_quantiles, 'o', 'MarkerSize', 4);
hold on;
plot([0, 1], [0, 1], 'r--', 'LineWidth', 1.5);
xlabel('Theoretical Quantiles');
ylabel('Sample Quantiles');
title('Q-Q Plot');
grid on;

fprintf('\nSimulation completed successfully!\n'); 