import lmfit, math, glob, os
import numpy as np
import matplotlib.pyplot as plt
from data_loaders import load_multistart

# -----------------------#
# Single spectrum fit
# -----------------------#

def chi_square_statistics(results):
    # Extract chi-square values
    chi2_values = []
    for result in results:
        chi2_values.append(result.chisqr)
    chi2_array = np.array(chi2_values)
    # Calculate statistics
    mean_chi2 = np.mean(chi2_array)
    std_chi2 = np.std(chi2_array)
    median_chi2 = np.median(chi2_array)
    min_chi2 = np.min(chi2_array)
    max_chi2 = np.max(chi2_array)
    # Coefficient of Variation
    cv = std_chi2 / mean_chi2 if mean_chi2 != 0 else np.inf
    # Additional metrics
    range_chi2 = max_chi2 - min_chi2
    iqr_chi2 = np.percentile(chi2_array, 75) - np.percentile(chi2_array, 25)
    # Quality assessment
    # Fits within 2x of minimum are considered "consistent"
    consistent_threshold = 2 * min_chi2
    consistent_fits = np.sum(chi2_array <= consistent_threshold)
    consistency_ratio = consistent_fits / len(chi2_array)
    # Results dictionary
    stats = {
        'total_trials': len(results),
        'chi2_values': chi2_array,
        'mean': mean_chi2,
        'std': std_chi2,
        'median': median_chi2,
        'min': min_chi2,
        'max': max_chi2,
        'coefficient_of_variation': cv,
        'range': range_chi2,
        'iqr': iqr_chi2,
        'consistent_fits': consistent_fits,
        'consistency_ratio': consistency_ratio,
        'best_result_index': np.argmin(chi2_array)
    }
    return stats

def print_chi_square_analysis(stats):
    """
    Print a formatted report of chi-square statistics
    """
    if stats is None:
        return
        
    print("="*60)
    print("           CHI-SQUARE ANALYSIS REPORT")
    print("="*60)
    print(f"Total trials:           {stats['total_trials']}")
    print()
    print("Chi-Square Statistics:")
    print(f"  Mean:                 {stats['mean']:.4e}")
    print(f"  Std:                  {stats['std']:.4e}")
    print(f"  Median:               {stats['median']:.4e}")
    print(f"  Min:                  {stats['min']:.4e}")
    print(f"  Max:                  {stats['max']:.4e}")
    print(f"  Range:                {stats['range']:.4e}")
    print(f"  IQR:                  {stats['iqr']:.4e}")
    print()
    print("Consistency Analysis:")
    print(f"  Coefficient of Variation: {stats['coefficient_of_variation']:.4f}")
    print(f"  Consistency ratio:        {stats['consistency_ratio']:.2%}")
    print()
    
    # Quality assessment
    if stats['coefficient_of_variation'] < 0.05:
        print("✅ EXCELLENT: Very consistent results (CV < 5%)")
    elif stats['coefficient_of_variation'] < 0.10:
        print("✅ GOOD: Reasonably consistent results (CV < 10%)")
    elif stats['coefficient_of_variation'] < 0.20:
        print("⚠️  FAIR: Some variability in results (CV < 20%)")
    else:
        print("❌ POOR: High variability - possible multiple minima (CV ≥ 20%)")
    
    if stats['consistency_ratio'] > 0.80:
        print("✅ HIGH consistency ratio - fits converge to same minimum")
    elif stats['consistency_ratio'] > 0.60:
        print("⚠️  MODERATE consistency ratio - some scatter in solutions")
    else:
        print("❌ LOW consistency ratio - possible optimization issues")
    
    print(f"\n🏆 Best fit: Trial {stats['best_result_index']+1} (χ² = {stats['min']:.4e})")
    print("="*60)

def visualize_chi_square_distribution(stats):
    """
    Create visualizations of chi-square distribution
    """
    if stats is None or len(stats['chi2_values']) == 0:
        return
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histogram
    axes[0].hist(stats['chi2_values'], bins=min(10, len(stats['chi2_values'])), 
                 alpha=0.7, edgecolor='black')
    axes[0].axvline(stats['mean'], color='red', linestyle='--', 
                    label=f'Mean: {stats["mean"]:.2e}')
    axes[0].axvline(stats['median'], color='green', linestyle='--', 
                    label=f'Median: {stats["median"]:.2e}')
    axes[0].set_xlabel('Chi-Square Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of χ² Values')
    axes[0].legend()
    axes[0].set_yscale('log') if stats['max'] / stats['min'] > 100 else None
    
    # Box plot
    axes[1].boxplot(stats['chi2_values'])
    axes[1].set_ylabel('Chi-Square Value')
    axes[1].set_title('χ² Box Plot')
    axes[1].set_yscale('log') if stats['max'] / stats['min'] > 100 else None
    
    # Trial sequence
    axes[2].plot(range(1, len(stats['chi2_values'])+1), stats['chi2_values'], 'o-')
    axes[2].axhline(stats['mean'], color='red', linestyle='--', alpha=0.7)
    axes[2].axhline(2*stats['min'], color='orange', linestyle=':', 
                    label='2×min threshold')
    axes[2].set_xlabel('Trial Number')
    axes[2].set_ylabel('Chi-Square Value')
    axes[2].set_title('χ² vs Trial Number')
    axes[2].legend()
    axes[2].set_yscale('log') if stats['max'] / stats['min'] > 100 else None
    
    plt.tight_layout()
    plt.show()


def analyze_fitting_consistency(saving_folder, cycles_to_test, sequences_to_test, reg_factors):
    """
    Analyze consistency of multistart fittings across all parameter combinations.
    Automatically discovers available spectrum indices for each combination.
    
    Parameters:
    - saving_folder: base folder containing results
    - cycles_to_test, sequences_to_test, reg_factors: parameter lists used in fitting
    
    Returns:
    - inconsistent_fits: list of parameter combinations with poor consistency
    - all_stats: dictionary with statistics for all combinations
    """
    
    inconsistent_fits = []
    all_stats = {}
    total_combinations = 0
    problematic_combinations = 0
    
    print("Analyzing fitting consistency across all parameter combinations...")
    print("="*80)
    
    for c in cycles_to_test:
        for s in sequences_to_test:
            # First level: cycle_sequence
            cycle_sequence_path = f"{saving_folder}/cycle{c}_sequence{s}"
            
            if not os.path.exists(cycle_sequence_path):
                print(f"⚠️  Path not found: {cycle_sequence_path}")
                continue
            
            # Find all spectrum folders
            spectrum_pattern = os.path.join(cycle_sequence_path, "spectrum*")
            spectrum_paths = glob.glob(spectrum_pattern)
            
            for spectrum_path in spectrum_paths:
                # Extract spectrum index from folder name
                spectrum_folder = os.path.basename(spectrum_path)
                i = int(spectrum_folder.replace("spectrum", ""))
                
                for r in reg_factors:
                    # Final path: spectrum -> reg_factor (using same format as saving)
                    result_path = os.path.join(spectrum_path, f"reg_factor{r:1.0e}")
                    
                    if not os.path.exists(result_path):
                        print(f"⚠️  Path not found: {result_path}")
                        continue
                    
                    total_combinations += 1
                    combination_key = f"c{c}_s{s}_r{r:1.0e}_i{i:03d}"
                    
                    try:
                        # Load multistart results
                        model, results, impedance, fits = load_multistart(result_path)
                        
                        # Analyze consistency
                        stats = chi_square_statistics(results)
                        all_stats[combination_key] = stats
                        
                        # Check for inconsistency (you can adjust these thresholds)
                        is_inconsistent = (
                            stats['coefficient_of_variation'] > 0.20 or  # High variability
                            stats['consistency_ratio'] < 0.60           # Low consistency ratio
                        )
                        
                        if is_inconsistent:
                            inconsistent_fits.append({
                                'cycle': c,
                                'sequence': s,
                                'reg_factor': r,
                                'spectrum': i,
                                'path': result_path,
                                'cv': stats['coefficient_of_variation'],
                                'consistency_ratio': stats['consistency_ratio'],
                                'stats': stats
                            })
                            problematic_combinations += 1
                            
                            print(f"⚠️  INCONSISTENT: {combination_key}")
                            print(f"    CV: {stats['coefficient_of_variation']:.4f}, Consistency: {stats['consistency_ratio']:.2%}")
                        
                    except Exception as e:
                        print(f"❌ ERROR loading {result_path}: {e}")
                        inconsistent_fits.append({
                            'cycle': c, 'sequence': s, 'reg_factor': r, 'spectrum': i,
                            'path': result_path, 'error': str(e)
                        })
                        problematic_combinations += 1
    
    # Summary report
    print("\n" + "="*80)
    print("                        CONSISTENCY ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total combinations analyzed:     {total_combinations}")
    print(f"Problematic combinations:        {problematic_combinations}")
    print(f"Success rate:                    {(total_combinations-problematic_combinations)/total_combinations:.1%}")
    
    if inconsistent_fits:
        print(f"\n❌ Found {len(inconsistent_fits)} combinations with poor consistency!")
        print("   This suggests the objective function is not sufficiently convex.")
        print("   Starting parameters matter for these cases.")
        
        # Group by regularization factor for detailed analysis
        reg_factor_details = {}
        for fit in inconsistent_fits:
            if 'reg_factor' in fit and 'error' not in fit:  # Only valid fits
                r = fit['reg_factor']
                if r not in reg_factor_details:
                    reg_factor_details[r] = []
                reg_factor_details[r].append(fit)
        
        print("\n" + "="*60)
        print("DETAILED INCONSISTENCY ANALYSIS BY REGULARIZATION FACTOR:")
        print("="*60)
        
        for r in sorted(reg_factor_details.keys()):
            fits_list = reg_factor_details[r]
            print(f"\n📊 reg_factor {r:1.0e}: {len(fits_list)} problematic cases")
            print("-" * 50)
            
            # Sort by consistency ratio (worst first)
            fits_list.sort(key=lambda x: x['consistency_ratio'])
            
            for fit in fits_list:
                consistency_pct = fit['consistency_ratio'] * 100
                cv = fit['cv']
                
                # Determine severity
                if consistency_pct < 30:
                    severity = "🔴 SEVERE"
                elif consistency_pct < 50:
                    severity = "🟠 HIGH"
                else:
                    severity = "🟡 MODERATE"
                    
                print(f"  {severity} | cycle={fit['cycle']}, sequence={fit['sequence']}, "
                      f"spectrum={fit['spectrum']:03d} | "
                      f"Consistency: {consistency_pct:.1f}%, CV: {cv:.3f}")
        
        # Show error cases separately
        error_cases = [fit for fit in inconsistent_fits if 'error' in fit]
        if error_cases:
            print(f"\n💥 ERROR CASES ({len(error_cases)}):")
            print("-" * 50)
            for fit in error_cases:
                print(f"  ❌ cycle={fit['cycle']}, sequence={fit['sequence']}, "
                      f"spectrum={fit['spectrum']:03d}, reg_factor={fit['reg_factor']:1.0e}")
                print(f"     Error: {fit['error']}")
            
    else:
        print("\n✅ All fittings show good consistency!")
        print("   The objective function appears sufficiently convex.")
    
    return inconsistent_fits, all_stats

# -----------------------#
# Simultanous spectra fit
# -----------------------#

def calculate_smoothness_metrics(results, param_names):
    """Calculate smoothness metrics for parameter evolution"""
    smoothness_metrics = {}
    
    for param_name in param_names:
        param_series = [result.params[param_name].value for result in results]
        param_series = np.array(param_series)
        
        # First derivative (gradient) smoothness
        if len(param_series) > 1:
            first_deriv = np.diff(param_series)
            gradient_smoothness = np.sum(first_deriv**2)
        else:
            gradient_smoothness = 0
        
        # Second derivative (curvature) smoothness
        if len(param_series) > 2:
            second_deriv = param_series[2:] - 2*param_series[1:-1] + param_series[:-2]
            curvature_smoothness = np.sum(second_deriv**2)
        else:
            curvature_smoothness = 0
        
        smoothness_metrics[param_name] = {
            'gradient': gradient_smoothness,
            'curvature': curvature_smoothness
        }
    
    return smoothness_metrics

def extract_global_param_evolution(global_result, param_names, n_spectra):
    """Extract parameter evolution from global result"""
    param_evolution = {name: [] for name in param_names}
    
    for t in range(n_spectra):
        for param_name in param_names:
            param_key = f'{param_name}_t{t}'
            param_evolution[param_name].append(global_result.params[param_key].value)
    
    return param_evolution

def plot_comparison(sequential_results, global_param_evolution, sequential_fits, global_fits, 
                    model_fun):
    """Plot comparison between sequential and global fitting results"""
    
    param_names = model_fun.param_names

    cols = 5
    raws = math.ceil(len(param_names)/cols)
    
    # Parameter evolution comparison
    fig, axes = plt.subplots(raws, cols, figsize=(14, 8))
    axes = axes.flatten()
    
    for i, param_name in enumerate(param_names):
        ax = axes[i]
        
        # Sequential results
        sequential_values = [result.params[param_name].value for result in sequential_results]
        ax.plot(sequential_values, 'b-o', label='Sequential', markersize=3, linewidth=1)
        
        # Global results
        global_values = global_param_evolution[param_name]
        ax.plot(global_values, 'r-s', label='Global+Smoothing', markersize=3, linewidth=1)
        
        ax.set_title(f'Parameter {param_name}')
        ax.set_xlabel('Time Point')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    if len(param_names) < len(axes):
        axes[-1].remove()
    
    plt.tight_layout()
    
    # Calculate and display smoothness metrics
    sequential_smoothness = calculate_smoothness_metrics(sequential_results, param_names)
    
    # Create pseudo-results for global parameters
    global_pseudo_results = []
    for t in range(len(sequential_results)):
        pseudo_result = type('obj', (object,), {})()
        pseudo_result.params = lmfit.Parameters()
        for param_name in param_names:
            pseudo_result.params.add(param_name, value=global_param_evolution[param_name][t])
        global_pseudo_results.append(pseudo_result)
    
    global_smoothness = calculate_smoothness_metrics(global_pseudo_results, param_names)
    
    print("\n" + "="*60)
    print("SMOOTHNESS COMPARISON")
    print("="*60)
    print(f"{'Parameter':<8} {'Sequential':<20} {'Global+Smoothing':<20} {'Difference':<12}")
    print("-" * 60)
    
    for param_name in param_names:
        seq_curv = sequential_smoothness[param_name]['curvature']
        glob_curv = global_smoothness[param_name]['curvature']
        improvement = seq_curv / glob_curv if glob_curv > 0 else float('inf')
        
        print(f"{param_name:<8} {seq_curv:<20.2e} {glob_curv:<20.2e} {improvement:<12.1f}x")
    
    plt.show()