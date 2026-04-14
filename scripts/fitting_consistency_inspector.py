from fit_statistics import analyze_fitting_consistency

saving_folder =  "experiments/2510070920multiple_starting_points_cell10_ratpoly6/data"
cycles_to_test = [1, 5, 10, 15, 20]
sequences_to_test = [0, 2, 4]  
reg_factors = [1e-5, 1e-4, 1e-3, 1e-2]

inconsistent_fits, all_stats = analyze_fitting_consistency(
    saving_folder, 
    cycles_to_test, 
    sequences_to_test, 
    reg_factors,
)