import json
import lmfit
import numpy as np
import os
from itertools import product
import pandas as pd
from datetime import datetime
import traceback

from routines import multiple_start_rational_poly_fit
from rational_poly_models import rational_poly5, rational_poly6, rational_poly7
from obj_functions import complex_least_square_regl1
import fit_statistics
from data_savers import save_multistart



# Define parameter space
experiment_folder = "C:/Users/fscarp/ownCloud/Data/2025/Coin_MnO2_2025/PanCoin2025_010/2507281144PanCoin2025_010_deisCCCV_x30_100s_window_fmin10Hz_fs_50Hz"

# Define model
model = lmfit.Model(rational_poly7)

# Test parameters
cycles_to_test = [1, 5, 10, 15, 20]
sequences_to_test = [0, 2, 4]  

# Spectra selection parameters
spectra_selection = {
    'start_freq': 1,
    'end_freq': -1,
    'start_col': 2,
    'end_col': -2,
    'step': 30   
}

# Regularization factors
reg_factors = [1e-5, 1e-4, 1e-3, 1e-2]

# Fitting parameters
n_trials = 50
seed = 42

# Saving
saving_folder =  "experiments/2510071441multiple_starting_points_cell10_ratpoly7/data"

print('Process started.')
for c in cycles_to_test:
    # Load frequencies
    metadata_file = open(experiment_folder + "/metadata_deis_exp.json")
    frequencies = np.array( json.load(metadata_file)['Frequencies multisine (Hz)'] )[spectra_selection['start_freq']:spectra_selection['end_freq']]
    for s in sequences_to_test:
        # Load data
        folder = experiment_folder + f'/pico_aquisition/cycle_{c}_sequence_{s}'
        impedance = np.load(folder+'/impedance_total.npy')[spectra_selection['start_freq']:spectra_selection['end_freq'],spectra_selection['start_col']:spectra_selection['end_col']:spectra_selection['step']]
        for i in range(impedance.shape[1]):
            for r in reg_factors:
                print(f"Processing: cycle={c}, sequence={s}, spectrum={i}, reg_factor={r}")
                # Fit
                results, fits = multiple_start_rational_poly_fit(
                    frequencies,
                    impedance[:,i],
                    model,
                    complex_least_square_regl1,
                    {'reg_factor':r},
                    n_trials=50,
                    seed=40,
                    min = 1e-6,
                    max = 1e4
                )
                saving_path = saving_folder+f"/cycle{c}_sequence{s}/spectrum{i:3d}/reg_factor{r:1.0e}"
                save_multistart(
                    saving_path,
                    model,
                    results,
                    impedance[:,i],
                    fits, 
                )
print('Process terminated.')













# def systematic_impedance_screening():
#     """
#     Systematic screening of impedance fitting parameters with multiple spectra per sequence
#     """
    
#     # Define parameter space
#     base_data_path = 'C:/Users/fscarp/ownCloud/Data/2025/Coin_MnO2_2025'
    
#     # Test parameters
#     cells_to_test = ['PanCoin2025_010']
#     cycles_to_test = [1, 5, 10, 15, 20]
#     sequences_to_test = [0, 1]  # 0=charge, 1=discharge
    
#     # Spectra selection parameters
#     spectra_selection = {
#         'start_col': 1,      # Start from column 1
#         'end_col': -1,       # End at second-to-last column  
#         'step': 30           # Every 30th spectrum
#     }
    
#     # Model parameters
#     models_config = {
#         'rational_poly5': {'function': rational_poly5, 'order': 5},
#         'rational_poly6': {'function': rational_poly6, 'order': 6},
#         'rational_poly7': {'function': rational_poly7, 'order': 7}
#     }
    
#     # Regularization factors
#     reg_factors = np.logspace(-5, -2, 4)
    
#     # Fitting parameters
#     n_trials = 50
#     seed = 40
    
#     # Results tracking
#     results_summary = []
#     base_save_path = f"./data/systematic_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#     os.makedirs(base_save_path, exist_ok=True)
    
#     print(f"Starting systematic screening with multiple spectra...")
#     print("="*60)
    
#     # Main screening loops
#     for cell_id in cells_to_test:
#         cell_path = find_cell_path(base_data_path, cell_id)
#         if not cell_path:
#             print(f"Cell {cell_id} not found, skipping...")
#             continue
            
#         for cycle in cycles_to_test:
#             for sequence in sequences_to_test:
                
#                 # Load all spectra for this sequence
#                 try:
#                     impedance_set, frequencies = load_multiple_impedance_data(
#                         cell_path, cycle, sequence, spectra_selection)
                    
#                     if impedance_set is None:
#                         print(f"Data not found for {cell_id} cycle {cycle} sequence {sequence}, skipping...")
#                         continue
                        
#                     n_spectra = len(impedance_set)
#                     print(f"\nProcessing {cell_id} cycle {cycle} sequence {sequence}: {n_spectra} spectra")
                    
#                 except Exception as e:
#                     print(f"Error loading data for {cell_id} cycle {cycle} sequence {sequence}: {e}")
#                     continue
                
#                 for model_name, model_config in models_config.items():
#                     for reg_factor in reg_factors:
                        
#                         combination_base_id = f"{cell_id}_cycle{cycle}_seq{sequence}_{model_name}_reg{reg_factor:.1e}"
#                         print(f"  Testing {model_name} with reg_factor {reg_factor:.1e}")
                        
#                         # Results for this parameter combination across all spectra
#                         combination_results = []
#                         combination_save_path = os.path.join(
#                             base_save_path, cell_id, f"cycle_{cycle}", 
#                             f"sequence_{sequence}", model_name, f"reg_{reg_factor:.1e}"
#                         )
                        
#                         # Fit each spectrum in the sequence
#                         for spectrum_idx, impedance_data in enumerate(impedance_set):
                            
#                             spectrum_id = f"{combination_base_id}_spectrum{spectrum_idx:03d}"
                            
#                             try:
#                                 # Perform fitting for this spectrum
#                                 model = lmfit.Model(model_config['function'])
#                                 obj_args = {'reg_factor': reg_factor}
                                
#                                 results, fits = multiple_start_rational_poly_fit(
#                                     frequencies,
#                                     impedance_data,
#                                     model,
#                                     complex_least_square_regl1,
#                                     obj_args,
#                                     n_trials=n_trials,
#                                     seed=seed + spectrum_idx  # Different seed per spectrum
#                                 )
                                
#                                 # Calculate statistics
#                                 stats = fit_statistics.chi_square_statistics(results)
                                
#                                 # Save results for this spectrum
#                                 spectrum_save_path = os.path.join(combination_save_path, f"spectrum_{spectrum_idx:03d}")
#                                 save_multistart(spectrum_save_path, model, results, impedance_data, fits)
                                
#                                 # Save spectrum metadata
#                                 save_spectrum_metadata(spectrum_save_path, {
#                                     'cell_id': cell_id,
#                                     'cycle': cycle,
#                                     'sequence': sequence,
#                                     'spectrum_index': spectrum_idx,
#                                     'total_spectra': n_spectra,
#                                     'model_name': model_name,
#                                     'model_order': model_config['order'],
#                                     'reg_factor': reg_factor,
#                                     'n_trials': n_trials,
#                                     'seed': seed + spectrum_idx,
#                                     'data_points': len(impedance_data),
#                                     'frequency_range': [float(frequencies.min()), float(frequencies.max())]
#                                 })
                                
#                                 # Collect summary statistics for this spectrum
#                                 best_result = min(results, key=lambda x: x.chisqr)
#                                 spectrum_summary = {
#                                     'spectrum_id': spectrum_id,
#                                     'combination_base_id': combination_base_id,
#                                     'cell_id': cell_id,
#                                     'cycle': cycle,
#                                     'sequence': sequence,
#                                     'spectrum_index': spectrum_idx,
#                                     'model_name': model_name,
#                                     'model_order': model_config['order'],
#                                     'reg_factor': reg_factor,
#                                     'best_chisqr': best_result.chisqr,
#                                     'best_aic': best_result.aic,
#                                     'best_bic': best_result.bic,
#                                     'n_successful_fits': len([r for r in results if r.success]),
#                                     'n_total_trials': len(results),
#                                     'save_path': spectrum_save_path,
#                                     'status': 'success'
#                                 }
                                
#                                 combination_results.append(spectrum_summary)
#                                 results_summary.append(spectrum_summary)
                                
#                                 if spectrum_idx % 10 == 0:  # Progress update every 10 spectra
#                                     print(f"    Spectrum {spectrum_idx+1}/{n_spectra} - Chi²: {best_result.chisqr:.2e}")
                                
#                             except Exception as e:
#                                 print(f"    ❌ Error fitting spectrum {spectrum_idx}: {e}")
                                
#                                 # Record failed attempt
#                                 spectrum_summary = {
#                                     'spectrum_id': spectrum_id,
#                                     'combination_base_id': combination_base_id,
#                                     'cell_id': cell_id,
#                                     'cycle': cycle,
#                                     'sequence': sequence,
#                                     'spectrum_index': spectrum_idx,
#                                     'model_name': model_name,
#                                     'model_order': model_config['order'],
#                                     'reg_factor': reg_factor,
#                                     'best_chisqr': np.nan,
#                                     'best_aic': np.nan,
#                                     'best_bic': np.nan,
#                                     'n_successful_fits': 0,
#                                     'n_total_trials': 0,
#                                     'save_path': '',
#                                     'status': 'failed',
#                                     'error': str(e)
#                                 }
#                                 results_summary.append(spectrum_summary)
                        
#                         # Save combination-level summary
#                         save_combination_summary(combination_save_path, combination_results)
                        
#                         # Print combination summary
#                         successful_spectra = [r for r in combination_results if r['status'] == 'success']
#                         if successful_spectra:
#                             avg_aic = np.mean([r['best_aic'] for r in successful_spectra])
#                             print(f"    ✅ {len(successful_spectra)}/{n_spectra} successful fits, avg AIC: {avg_aic:.2f}")
#                         else:
#                             print(f"    ❌ No successful fits for this combination")
    
#     # Save overall summary
#     summary_df = pd.DataFrame(results_summary)
#     summary_path = os.path.join(base_save_path, 'screening_summary.csv')
#     summary_df.to_csv(summary_path, index=False)
    
#     # Save aggregated statistics
#     save_aggregated_statistics(base_save_path, summary_df)
    
#     print("="*60)
#     print("Screening completed!")
#     print(f"Results saved to: {base_save_path}")
#     print(f"Summary saved to: {summary_path}")
    
#     # Print analysis
#     analyze_screening_results(summary_df)
    
#     return summary_df, base_save_path

# def find_cell_path(base_path, cell_id):
#     """Find the full path to a cell's data"""
#     for root, dirs, files in os.walk(base_path):
#         if cell_id in root:
#             return root
#     return None

# def load_multiple_impedance_data(cell_path, cycle, sequence, spectra_selection):
#     """Load multiple impedance spectra for specific cycle and sequence"""
#     try:
#         # Find data folder
#         data_folder = None
#         for root, dirs, files in os.walk(cell_path):
#             if f'cycle_{cycle}_sequence_{sequence}' in root:
#                 data_folder = root
#                 break
        
#         if not data_folder:
#             return None, None
        
#         # Load impedance data
#         impedance_file = os.path.join(data_folder, 'impedance_total.npy')
#         if not os.path.exists(impedance_file):
#             return None, None
            
#         # Load all impedance data
#         impedance_full = np.load(impedance_file)
        
#         # Extract multiple spectra using the selection parameters
#         start_col = spectra_selection['start_col']
#         end_col = spectra_selection['end_col']
#         step = spectra_selection['step']
        
#         # Select spectra: [frequency_points, spectra_selection]
#         impedance_selected = impedance_full[1:, start_col:end_col:step]
        
#         # Convert to list of individual spectra
#         impedance_set = [impedance_selected[:, i] for i in range(impedance_selected.shape[1])]
        
#         # Load frequencies
#         metadata_file = os.path.join(cell_path, 'metadata_deis_exp.json')
#         with open(metadata_file) as f:
#             metadata = json.load(f)
#         frequencies = np.array(metadata['Frequencies multisine (Hz)'])[1:]
        
#         return impedance_set, frequencies
        
#     except Exception as e:
#         print(f"Error loading multiple impedance data: {e}")
#         return None, None

# def save_spectrum_metadata(save_path, metadata):
#     """Save metadata for individual spectrum"""
#     os.makedirs(save_path, exist_ok=True)
#     metadata_path = os.path.join(save_path, 'spectrum_metadata.json')
#     with open(metadata_path, 'w') as f:
#         json.dump(metadata, f, indent=2)

# def save_combination_summary(save_path, combination_results):
#     """Save summary for a parameter combination across all spectra"""
#     os.makedirs(save_path, exist_ok=True)
    
#     # Convert to DataFrame for easier analysis
#     combo_df = pd.DataFrame(combination_results)
    
#     # Summary statistics
#     successful_fits = combo_df[combo_df['status'] == 'success']
    
#     summary_stats = {
#         'total_spectra': len(combination_results),
#         'successful_fits': len(successful_fits),
#         'success_rate': len(successful_fits) / len(combination_results) if combination_results else 0,
#         'avg_aic': float(successful_fits['best_aic'].mean()) if len(successful_fits) > 0 else np.nan,
#         'std_aic': float(successful_fits['best_aic'].std()) if len(successful_fits) > 0 else np.nan,
#         'avg_chisqr': float(successful_fits['best_chisqr'].mean()) if len(successful_fits) > 0 else np.nan,
#         'median_aic': float(successful_fits['best_aic'].median()) if len(successful_fits) > 0 else np.nan,
#         'best_spectrum_aic': float(successful_fits['best_aic'].min()) if len(successful_fits) > 0 else np.nan,
#         'worst_spectrum_aic': float(successful_fits['best_aic'].max()) if len(successful_fits) > 0 else np.nan
#     }
    
#     # Save combination summary
#     combo_summary_path = os.path.join(save_path, 'combination_summary.json')
#     with open(combo_summary_path, 'w') as f:
#         json.dump(summary_stats, f, indent=2)
    
#     # Save detailed results
#     combo_details_path = os.path.join(save_path, 'spectra_results.csv')
#     combo_df.to_csv(combo_details_path, index=False)

# def save_aggregated_statistics(base_save_path, summary_df):
#     """Save aggregated statistics across all combinations with error handling"""
    
#     # Debug: Check what's in the DataFrame
#     print(f"\nDebugging save_aggregated_statistics:")
#     print(f"DataFrame shape: {summary_df.shape}")
#     print(f"DataFrame columns: {list(summary_df.columns)}")
    
#     if summary_df.empty:
#         print("Warning: summary_df is empty, skipping aggregated statistics")
#         return None
    
#     # Check for required columns
#     required_cols = ['cell_id', 'cycle', 'sequence', 'model_name', 'reg_factor']
#     missing_cols = [col for col in required_cols if col not in summary_df.columns]
    
#     if missing_cols:
#         print(f"Warning: Missing columns {missing_cols} in summary_df")
#         print("Available columns:", list(summary_df.columns))
#         return None
    
#     # Filter successful fits only for aggregation
#     successful_fits = summary_df[summary_df['status'] == 'success']
    
#     if successful_fits.empty:
#         print("Warning: No successful fits found for aggregation")
#         # Still save the empty results
#         empty_stats = pd.DataFrame()
#         agg_path = os.path.join(base_save_path, 'aggregated_statistics.csv')
#         empty_stats.to_csv(agg_path, index=False)
#         return empty_stats
    
#     print(f"Aggregating {len(successful_fits)} successful fits...")
    
#     try:
#         # Group by parameter combinations
#         grouping_cols = ['cell_id', 'cycle', 'sequence', 'model_name', 'reg_factor']
        
#         # Check if we have the required columns for aggregation
#         agg_columns = {}
#         if 'best_aic' in successful_fits.columns:
#             agg_columns['best_aic'] = ['count', 'mean', 'std', 'median', 'min', 'max']
#         if 'best_chisqr' in successful_fits.columns:
#             agg_columns['best_chisqr'] = ['mean', 'std', 'median']
#         if 'n_successful_fits' in successful_fits.columns:
#             agg_columns['n_successful_fits'] = 'sum'
#         if 'n_total_trials' in successful_fits.columns:
#             agg_columns['n_total_trials'] = 'sum'
        
#         if not agg_columns:
#             print("Warning: No aggregation columns found")
#             return None
            
#         agg_stats = successful_fits.groupby(grouping_cols).agg(agg_columns).round(4)
        
#         # Flatten column names
#         agg_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
#                             for col in agg_stats.columns.values]
#         agg_stats = agg_stats.reset_index()
        
#         # Add success rate if possible
#         if 'n_successful_fits_sum' in agg_stats.columns and 'n_total_trials_sum' in agg_stats.columns:
#             agg_stats['success_rate'] = agg_stats['n_successful_fits_sum'] / agg_stats['n_total_trials_sum']
        
#         # Save aggregated results
#         agg_path = os.path.join(base_save_path, 'aggregated_statistics.csv')
#         agg_stats.to_csv(agg_path, index=False)
        
#         print(f"Aggregated statistics saved to: {agg_path}")
#         print(f"Aggregated {len(agg_stats)} parameter combinations")
        
#         return agg_stats
        
#     except Exception as e:
#         print(f"Error in aggregation: {e}")
#         print("Saving individual results only...")
        
#         # Save the successful fits as backup
#         backup_path = os.path.join(base_save_path, 'successful_fits_backup.csv')
#         successful_fits.to_csv(backup_path, index=False)
        
#         return None

# def analyze_screening_results(summary_df):
#     """Analysis of screening results with multiple spectra"""
#     successful_fits = summary_df[summary_df['status'] == 'success']
    
#     if len(successful_fits) == 0:
#         print("No successful fits found!")
#         return
    
#     print(f"\nOverall Analysis:")
#     print(f"Total spectra fitted: {len(summary_df)}")
#     print(f"Successful fits: {len(successful_fits)} ({100*len(successful_fits)/len(summary_df):.1f}%)")
    
#     # Best spectrum overall
#     best_spectrum = successful_fits.loc[successful_fits['best_aic'].idxmin()]
#     print(f"\nBest spectrum (lowest AIC):")
#     print(f"  ID: {best_spectrum['spectrum_id']}")
#     print(f"  AIC: {best_spectrum['best_aic']:.2f}")
#     print(f"  Chi²: {best_spectrum['best_chisqr']:.2e}")
    
#     # Best combination (averaged across spectra)
#     combo_stats = successful_fits.groupby(['cell_id', 'cycle', 'sequence', 'model_name', 'reg_factor'])['best_aic'].agg(['count', 'mean', 'std']).reset_index()
#     combo_stats = combo_stats[combo_stats['count'] >= 5]  # At least 5 successful fits
    
#     if len(combo_stats) > 0:
#         best_combo = combo_stats.loc[combo_stats['mean'].idxmin()]
#         print(f"\nBest parameter combination (avg AIC across spectra):")
#         print(f"  Model: {best_combo['model_name']}")
#         print(f"  Reg factor: {best_combo['reg_factor']:.1e}")
#         print(f"  Avg AIC: {best_combo['mean']:.2f} ± {best_combo['std']:.2f}")
#         print(f"  Successful spectra: {best_combo['count']}")

# # Usage remains the same
# if __name__ == "__main__":
#     summary_df, save_path = systematic_impedance_screening()