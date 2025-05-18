from FL.utilis.silo import Silo
from FL.utilis.server import Server
from FL.utilis.dataset import Dataset
from model.CNN import CNNMnist
import torch
import matplotlib.pyplot as plt
import numpy as np
# import logging # Commented out as we are switching to print
import os
from FL.utilis.hist_noise import histogram_dp, histogram_structure_first_dp, histogram_noise_first_dp

# logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(message)s', datefmt='%Y-%m-%d %H:%M')
# logger = logging.getLogger(__name__) # Switch to print for debugging parameter output

def plot_comparison_results(original_hist, hist_variants_data, output_dir, plot_file_basename, main_plot_title):
    """
    Generates and saves comparison plots for a set of histogram variants.

    Args:
        original_hist: The original histogram data.
        hist_variants_data: A list of tuples, where each tuple is 
                            (label, histogram_array, color_str).
        output_dir: Directory to save the plots.
        plot_file_basename: Base name for the output plot files (e.g., "nf_eps_0.5").
        main_plot_title: Title for the set of comparison plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    indices = np.arange(len(original_hist))
    num_variants = len(hist_variants_data)
    
    if num_variants == 0:
        print(f"WARNING: No variants provided for plotting: {plot_file_basename}. Skipping plots.")
        return

    # --- Plot 1: Side-by-side comparison of all variants ---
    plt.figure(figsize=(max(12, 3 * (num_variants + 1)), 8)) 
    
    total_bars_width = 0.8 
    width = total_bars_width / (num_variants + 1)

    plt.bar(indices - (num_variants/2.0)*width, original_hist, width, label='Original', color='blue', alpha=0.7)

    for i, (label, data, color) in enumerate(hist_variants_data):
        offset = (i - num_variants/2.0 + 1) * width
        plt.bar(indices + offset, data, width, label=label, color=color, alpha=0.7)

    plt.xlabel('Bin Index')
    plt.ylabel('Count')
    plt.title(f'Overall Comparison: {main_plot_title}')
    plt.xticks(indices)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1)) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    plt.savefig(f'{output_dir}/{plot_file_basename}_overall.png', dpi=300)
    plt.close()

    # --- Plot 2: Detailed subplots (Original vs each variant) & Differences ---
    
    gaussian_variant_info = None
    other_variants_info = []
    for item_tuple in hist_variants_data: # item_tuple is (label, data, color)
        if item_tuple[0].startswith("GaussianDP"):
            gaussian_variant_info = item_tuple
        else:
            other_variants_info.append(item_tuple)

    num_other_variants = len(other_variants_info)
    
    num_subplot_rows_detailed = (num_other_variants + 1) // 2
    if gaussian_variant_info or hist_variants_data: 
        num_subplot_rows_detailed += 1


    plt.figure(figsize=(20, 6 * num_subplot_rows_detailed))
    
    current_subplot_idx = 1
    for label, data, color in other_variants_info:
        plt.subplot(num_subplot_rows_detailed, 2, current_subplot_idx)
        plt.bar(indices, original_hist, alpha=0.7, label='Original', color='blue')
        plt.bar(indices, data, alpha=0.5, label=label, color=color)
        plt.title(f'Original vs {label}')
        plt.xlabel('Bin Index')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        current_subplot_idx += 1

    last_row_start_idx = current_subplot_idx

    if gaussian_variant_info:
        plt.subplot(num_subplot_rows_detailed, 2, last_row_start_idx)
        label_g, data_g, color_g = gaussian_variant_info
        plt.bar(indices, original_hist, alpha=0.7, label='Original', color='blue')
        plt.bar(indices, data_g, alpha=0.5, label=label_g, color=color_g)
        plt.title(f'Original vs {label_g}')
        plt.xlabel('Bin Index')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        diff_plot_idx = last_row_start_idx + 1
    else:
        diff_plot_idx = last_row_start_idx


    if hist_variants_data: 
        if diff_plot_idx <= num_subplot_rows_detailed * 2:
            plt.subplot(num_subplot_rows_detailed, 2, diff_plot_idx)
            diff_bar_width_total = 0.8
            num_all_variants_for_diff = len(hist_variants_data)
            diff_bar_width = diff_bar_width_total / num_all_variants_for_diff if num_all_variants_for_diff > 0 else diff_bar_width_total

            for i, (label_diff, data_diff, color_diff) in enumerate(hist_variants_data): 
                diff_values = np.array(data_diff) - np.array(original_hist)
                offset = (i - (num_all_variants_for_diff - 1) / 2.0) * diff_bar_width
                plt.bar(indices + offset, diff_values, diff_bar_width, label=f'{label_diff} Diff', color=color_diff, alpha=0.7)
            
            plt.title('Difference from Original Histogram')
            plt.xlabel('Bin Index')
            plt.ylabel('Difference')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            plt.legend(fontsize='small', loc='upper right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            print(f"WARNING: Could not place difference plot for {plot_file_basename} due to subplot indexing issues.")
    
    plt.suptitle(f'Detailed Comparison: {main_plot_title}', fontsize=16)
    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])
    plt.savefig(f'{output_dir}/{plot_file_basename}_detailed.png', dpi=300)
    plt.close()

    # --- Plot 3: Error Metrics (MAE & MSE) ---
    fig_metrics, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(15, 2.5 * num_variants), 8)) 
    
    method_labels = [item[0] for item in hist_variants_data]
    method_colors = [item[2] for item in hist_variants_data]
    
    mae_values = [np.mean(np.abs(np.array(item[1]) - np.array(original_hist))) for item in hist_variants_data]
    mse_values = [np.mean(np.square(np.array(item[1]) - np.array(original_hist))) for item in hist_variants_data]

    ax1.bar(method_labels, mae_values, color=method_colors, alpha=0.7)
    ax1.set_title('Mean Absolute Error')
    ax1.set_ylabel('MAE')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax2.bar(method_labels, mse_values, color=method_colors, alpha=0.7)
    ax2.set_title('Mean Squared Error')
    ax2.set_ylabel('MSE')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v_mae in enumerate(mae_values):
        ax1.text(i, v_mae + 0.01 * max(mae_values, default=1), f'{v_mae:.3f}', ha='center', va='bottom', fontsize=9)
    for i, v_mse in enumerate(mse_values):
        ax2.text(i, v_mse + 0.01 * max(mse_values, default=1), f'{v_mse:.3f}', ha='center', va='bottom', fontsize=9)
    
    fig_metrics.suptitle(f'Error Metrics: {main_plot_title}', fontsize=16)
    fig_metrics.tight_layout(pad=2.0, rect=[0, 0, 1, 0.95])
    plt.savefig(f'{output_dir}/{plot_file_basename}_error_metrics.png', dpi=300)
    plt.close(fig_metrics)
    print(f"INFO: Comparison plots saved to {output_dir} with basename {plot_file_basename}")


if __name__ == "__main__":
    dataset_name = "mnist"
    max_samples = 3000 
    batch_size = 64
    local_epoch = 10 
    distribute_mode = "zipf"
    is_secure = "ndp" 
    learning_rate = 0.1
    n_silos = 5 
    n_users = 100 
    seed = 1
    q_for_batch_size = 0.1
    is_histogram_secure = "ndp" 
    delta_fedavg = 1e-5 
    
    base_sigma_gaussian = 5.0 

    epsilons_to_test = [0.1, 0.5, 1.0, 5.0]
    Fs_to_test = [1.0, 10.0, 20.0]
    k_divisors_to_test = [15, 10, 5] 

    default_epsilon = 1.0
    default_F = 10.0
    default_k_divisor = 10

    model_cnn = CNNMnist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("INFO: Setting up dataset and silo to get a sample histogram...")
    dataset_obj = Dataset(
        dataset_name=dataset_name,
        n_silos=n_silos,
        n_users=n_users,
        seed=seed,
        mode=distribute_mode,
        max_samples=max_samples
    )
    server_obj = Server(dataset_obj, model_cnn, learning_rate, is_secure) 
    silo_instance = Silo(0, server_obj, dataset_obj, model_cnn, learning_rate, q_for_batch_size, local_epoch, is_secure, False, is_histogram_secure, delta_fedavg, base_sigma_gaussian, 1.0)
    
    if silo_instance.user_histogram is None or len(silo_instance.user_histogram) == 0:
        print("INFO: User histogram not yet available, performing a dummy update round...")
        silo_instance.silo_update() 
        if silo_instance.user_histogram is None or len(silo_instance.user_histogram) == 0:
            print("WARNING: Still no histogram, using a placeholder for testing.")
            sample_histogram = np.array([10,20,30,40,50,40,30,20,10,5,15,25,35,45,35,25,15,5])
        else:
            sample_histogram = silo_instance.user_histogram
    else:
        sample_histogram = silo_instance.user_histogram

    if len(sample_histogram) == 0: 
        print("ERROR: Failed to obtain or generate a valid sample_histogram. Exiting.")
        exit()
    print(f"INFO: Obtained sample histogram with {len(sample_histogram)} bins: {sample_histogram[:10]}...")

    main_output_dir = "FL/test/test_output/adv_histogram_comparisons"
    os.makedirs(main_output_dir, exist_ok=True)

    baseline_gaussian_hist = histogram_dp(sample_histogram, base_sigma_gaussian)
    gaussian_dp_variant_tuple = (f'GaussianDP σ={base_sigma_gaussian}', baseline_gaussian_hist, 'gray')

    # --- NoiseFirstDP: Epsilon Variation ---
    print("INFO: --- Running NoiseFirstDP Epsilon Variation Tests ---")
    nf_eps_output_dir = os.path.join(main_output_dir, "NoiseFirst_Epsilon_Var")
    current_variants_nf_eps = [gaussian_dp_variant_tuple] 
    colors_nf = ['mediumpurple', 'darkorchid', 'blueviolet', 'indigo']
    for i, eps in enumerate(epsilons_to_test):
        try:
            print(f"INFO: Running NoiseFirstDP (Epsilon Variation) with: ε={eps}, method='mean'")
            nf_mean_hist, best_k_m = histogram_noise_first_dp(sample_histogram, eps, method='mean')
            current_variants_nf_eps.append((f'NF Mean ε={eps} k={best_k_m}', nf_mean_hist, colors_nf[i % len(colors_nf)]))
            
            print(f"INFO: Running NoiseFirstDP (Epsilon Variation) with: ε={eps}, method='median'")
            nf_median_hist, best_k_d = histogram_noise_first_dp(sample_histogram, eps, method='median')
            current_variants_nf_eps.append((f'NF Median ε={eps} k={best_k_d}', nf_median_hist, plt.cm.viridis(i / len(epsilons_to_test) * 0.8 + 0.2) ))
        except Exception as e:
            print(f"ERROR: Error NF ε={eps}: {e}")
    plot_comparison_results(sample_histogram, current_variants_nf_eps, nf_eps_output_dir, "nf_epsilon_comp", "NoiseFirstDP Epsilon Variation")

    # --- StructureFirstDP: Epsilon Variation ---
    print("INFO: --- Running StructureFirstDP Epsilon Variation Tests ---")
    sf_eps_output_dir = os.path.join(main_output_dir, "StructureFirst_Epsilon_Var")
    k_for_sf_eps_var = max(1, int(len(sample_histogram) / default_k_divisor))
    current_variants_sf_eps = [gaussian_dp_variant_tuple]
    colors_sf_eps = ['lightcoral', 'red', 'darkred', 'maroon']
    for i, eps in enumerate(epsilons_to_test):
        try:
            # print(f"INFO: Running SF (Epsilon Var) with: ε={eps}, method='mean', k={k_for_sf_eps_var}, F={default_F}")
            sf_mean_hist = histogram_structure_first_dp(sample_histogram, eps, k_for_sf_eps_var, default_F, method='mean')
            current_variants_sf_eps.append((f'SF Mean ε={eps} F={default_F} k={k_for_sf_eps_var}', sf_mean_hist, colors_sf_eps[i % len(colors_sf_eps)]))
            # print(f"INFO: Running SF (Epsilon Var) with: ε={eps}, method='median', k={k_for_sf_eps_var}, F={default_F}")
            sf_median_hist = histogram_structure_first_dp(sample_histogram, eps, k_for_sf_eps_var, default_F, method='median')
            current_variants_sf_eps.append((f'SF Median ε={eps} F={default_F} k={k_for_sf_eps_var}', sf_median_hist, plt.cm.plasma(i / len(epsilons_to_test) * 0.8 + 0.1)))
        except Exception as e:
            print(f"ERROR: Error SF ε={eps} (F={default_F}, k={k_for_sf_eps_var}): {e}")
    plot_comparison_results(sample_histogram, current_variants_sf_eps, sf_eps_output_dir, "sf_epsilon_comp", f"SFDP Epsilon Var (F={default_F}, k_div={default_k_divisor})")

    # --- StructureFirstDP: F Variation ---
    print("INFO: --- Running StructureFirstDP F Variation Tests ---")
    sf_F_output_dir = os.path.join(main_output_dir, "StructureFirst_F_Var")
    k_for_sf_F_var = max(1, int(len(sample_histogram) / default_k_divisor))
    current_variants_sf_F = [gaussian_dp_variant_tuple]
    colors_sf_F = ['sandybrown', 'chocolate', 'saddlebrown']
    for i, F_val in enumerate(Fs_to_test):
        try:
            # print(f"INFO: Running SF (F Var) with: F={F_val}, method='mean', k={k_for_sf_F_var}, ε={default_epsilon}")
            sf_mean_hist = histogram_structure_first_dp(sample_histogram, default_epsilon, k_for_sf_F_var, F_val, method='mean')
            current_variants_sf_F.append((f'SF Mean F={F_val} ε={default_epsilon} k={k_for_sf_F_var}', sf_mean_hist, colors_sf_F[i % len(colors_sf_F)]))
            # print(f"INFO: Running SF (F Var) with: F={F_val}, method='median', k={k_for_sf_F_var}, ε={default_epsilon}")
            sf_median_hist = histogram_structure_first_dp(sample_histogram, default_epsilon, k_for_sf_F_var, F_val, method='median')
            current_variants_sf_F.append((f'SF Median F={F_val} ε={default_epsilon} k={k_for_sf_F_var}', sf_median_hist, plt.cm.YlOrBr(i / len(Fs_to_test) * 0.7 + 0.3)))
        except Exception as e:
            print(f"ERROR: Error SF F={F_val} (ε={default_epsilon}, k={k_for_sf_F_var}): {e}")
    plot_comparison_results(sample_histogram, current_variants_sf_F, sf_F_output_dir, "sf_F_comp", f"SFDP F Variation (ε={default_epsilon}, k_div={default_k_divisor})")

    # --- StructureFirstDP: k Variation ---
    print("INFO: --- Running StructureFirstDP k Variation Tests ---")
    sf_k_output_dir = os.path.join(main_output_dir, "StructureFirst_k_Var")
    current_variants_sf_k = [gaussian_dp_variant_tuple]
    colors_sf_k = ['lightgreen', 'limegreen', 'forestgreen']
    for i, k_div in enumerate(k_divisors_to_test):
        k_val = max(1, int(len(sample_histogram) / k_div))
        if k_val == 0 and len(sample_histogram) > 0 : k_val = 1 
        if k_val == 0 and len(sample_histogram) == 0:
            print(f"WARNING: Skipping k_div {k_div} as histogram is empty.")
            continue
        try:
            print(f"INFO: Running SF (k Var) with: k={k_val}, k_div={k_div}, method='mean', F={default_F}, ε={default_epsilon}")
            sf_mean_hist = histogram_structure_first_dp(sample_histogram, default_epsilon, k_val, default_F, method='mean')
            current_variants_sf_k.append((f'SF Mean k_div={k_div} (k={k_val}) ε={default_epsilon} F={default_F}', sf_mean_hist, colors_sf_k[i % len(colors_sf_k)])) 
            print(f"INFO: Running SF (k Var) with: k={k_val}, k_div={k_div}, method='median', F={default_F}, ε={default_epsilon}")
            sf_median_hist = histogram_structure_first_dp(sample_histogram, default_epsilon, k_val, default_F, method='median')
            current_variants_sf_k.append((f'SF Median k_div={k_div} (k={k_val}) ε={default_epsilon} F={default_F}', sf_median_hist, plt.cm.Greens(i / len(k_divisors_to_test) * 0.6 + 0.4))) 
        except Exception as e:
            print(f"ERROR: Error SF k_div={k_div} (ε={default_epsilon}, F={default_F}): {e}")
    plot_comparison_results(sample_histogram, current_variants_sf_k, sf_k_output_dir, "sf_k_comp", f"SFDP k Variation (ε={default_epsilon}, F={default_F})")
    
    # --- Final Overall Comparison ---
    print("INFO: --- Generating Final Overall Comparison ---")
    overall_comp_output_dir = os.path.join(main_output_dir, "Overall_Best_Comparison")
    
    all_representative_variants = [gaussian_dp_variant_tuple]

    try:
        print(f"INFO: Running NoiseFirstDP (Overall Comparison) with: ε={default_epsilon}, method='mean'")
        nf_mean_default_eps, best_k_m_def_eps = histogram_noise_first_dp(sample_histogram, default_epsilon, method='mean')
        all_representative_variants.append((f'NF Mean (default ε={default_epsilon}, k={best_k_m_def_eps})', nf_mean_default_eps, 'purple'))
        
        print(f"INFO: Running NoiseFirstDP (Overall Comparison) with: ε={default_epsilon}, method='median'")
        nf_median_default_eps, best_k_d_def_eps = histogram_noise_first_dp(sample_histogram, default_epsilon, method='median')
        all_representative_variants.append((f'NF Median (default ε={default_epsilon}, k={best_k_d_def_eps})', nf_median_default_eps, 'indigo'))
    except Exception as e:
        print(f"ERROR: Error generating representative NoiseFirst for overall comparison: {e}")

    k_for_overall_sf = max(1, int(len(sample_histogram) / default_k_divisor))
    try:
        # print(f"INFO: Running SF (Overall Comparison) with: default params, method='mean'")
        sf_mean_default_params = histogram_structure_first_dp(sample_histogram, default_epsilon, k_for_overall_sf, default_F, method='mean')
        all_representative_variants.append((f'SF Mean (default params: ε={default_epsilon}, F={default_F}, k_div={default_k_divisor}, k={k_for_overall_sf})', sf_mean_default_params, 'red'))
        # print(f"INFO: Running SF (Overall Comparison) with: default params, method='median'")
        sf_median_default_params = histogram_structure_first_dp(sample_histogram, default_epsilon, k_for_overall_sf, default_F, method='median')
        all_representative_variants.append((f'SF Median (default params: ε={default_epsilon}, F={default_F}, k_div={default_k_divisor}, k={k_for_overall_sf})', sf_median_default_params, 'darkred'))
    except Exception as e:
        print(f"ERROR: Error generating representative StructureFirst for overall comparison: {e}")
    
    plot_comparison_results(sample_histogram, all_representative_variants, overall_comp_output_dir, "overall_final_comp", "Overall Comparison of Representative Methods")

    print("INFO: All comparison tests finished. Plots saved in respective subdirectories of " + main_output_dir)
