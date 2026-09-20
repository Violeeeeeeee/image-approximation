import torch
import cv2
from ax.service.ax_client import AxClient, ObjectiveProperties

from utils.config_loader import load_config, parse_args
from modules.evolution_optimized import EvolutionarySolver
from modules.renderer_jit import render_batch

TARGET_TENSOR = None
GRID_X = None
GRID_Y = None
DEVICE = None
IMG_W = 0
IMG_H = 0

def train_evaluate(parameterization):
    """
    Executes a SINGLE experiment (trial) with parameters suggested by Ax.

    Args:
        parameterization (dict): Dictionary of hyperparameters selected by Ax
                                 (e.g., {'population_size': 128, 'mutation_rate': 0.01}).

    Returns:
        float: The final best MSE loss achieved in this trial (lower is better).
    """
    pop_size = int(parameterization.get('population_size', 512))
    n_triangles = int(parameterization.get('num_triangles', 50))
    mut_rate = parameterization.get('mutation_rate', 0.01)
    mut_scale = parameterization.get('mutation_scale', 15.0)
    survival_rate = parameterization.get('survival_rate', 0.1)

    # Create a temporary local configuration for this specific trial
    local_config = {
        'algorithm': {
            'population_size': pop_size,
            'num_triangles': n_triangles,
            'mutation_rate': mut_rate,
            'mutation_scale': mut_scale,
            'survival_rate': survival_rate
        }
    }

    # This sets up the random population tensor: [pop_size, num_triangles, 10]
    solver = EvolutionarySolver(local_config, IMG_W, IMG_H, DEVICE)

    # For hyperparameter tuning, we run fewer generations (e.g., 1500)
    # just to observe the convergence dynamic, saving time compared to a full run.
    N_GENERATIONS = 1500

    best_loss_final = float('inf')

    # Use global coordinate grids
    current_grid_x = GRID_X  # Shape: [1, 1, H, W]
    current_grid_y = GRID_Y  # Shape: [1, 1, H, W]

    # Fast training loop (using mininterval or no tqdm to keep logs clean)
    for gen in range(N_GENERATIONS):
        # --- Rendering Step ---
        # solver.genes shape: [Pop_Size, N_Triangles, 10]
        # rendered_pop shape: [Pop_Size, Height, Width, 3] (Channels Last)
        rendered_pop = render_batch(solver.genes, IMG_W, IMG_H, current_grid_x, current_grid_y)

        # --- Loss Calculation ---
        # TARGET_TENSOR shape: [1, Height, Width, 3]
        # diff shape: [Pop_Size, Height, Width, 3]
        diff = (rendered_pop - TARGET_TENSOR) ** 2

        # Calculate Mean Squared Error for each individual
        # mse_scores shape: [Pop_Size]
        mse_scores = diff.mean(dim=[1, 2, 3])

        min_loss = mse_scores.min().item()
        if min_loss < best_loss_final:
            best_loss_final = min_loss

        # --- Evolution Step ---
        # Performs selection, crossover, and mutation based on mse_scores
        solver.step(mse_scores)

        # --- Pruning (Optional) ---
        # If loss is still terrible at epoch 500, abort this trial to save time.
        if gen == 500 and min_loss > 0.05: # Threshold depends on the image complexity
             # Could raise a special exception or return early to tell Ax it's a bad path
             pass

    return best_loss_final

def main():
    global TARGET_TENSOR, GRID_X, GRID_Y, DEVICE, IMG_W, IMG_H

    args = parse_args()
    base_config = load_config(args.config)

    # --- SETUP RESOURCES ---
    # Enable cuDNN benchmarking for faster convolutions/matrix ops if shapes are constant
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    # --- LOAD DATA ---
    target_path = base_config['data']['target_path']
    target_img_cv = cv2.imread(target_path)
    target_img_cv = cv2.cvtColor(target_img_cv, cv2.COLOR_BGR2RGB)

    # FOR AX: Downscale image to 64x64 or 128x128 for faster search.
    # Once parameters are found, we can use them on the full 256x256 image.
    resize_to = 64
    target_img_cv = cv2.resize(target_img_cv, (resize_to, resize_to), interpolation=cv2.INTER_AREA)

    IMG_H, IMG_W, _ = target_img_cv.shape
    print(f"Optimization Resolution: {IMG_W}x{IMG_H}")

    # Prepare Target Tensor: Normalize to [0, 1]
    # Shape: [1, Height, Width, 3]
    TARGET_TENSOR = torch.from_numpy(target_img_cv).float().to(DEVICE) / 255.0
    TARGET_TENSOR = TARGET_TENSOR.unsqueeze(0)

    # --- PRE-CALCULATE GRIDS ---
    # We generate coordinate grids once to be reused by the JIT renderer.
    y_coords = torch.linspace(0, IMG_H - 1, IMG_H, device=DEVICE)
    x_coords = torch.linspace(0, IMG_W - 1, IMG_W, device=DEVICE)
    gy, gx = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Shape: [1, 1, Height, Width] - Expanded for broadcasting
    GRID_X = gx.unsqueeze(0).unsqueeze(0)
    GRID_Y = gy.unsqueeze(0).unsqueeze(0)

    # --- AX CONFIGURATION ---
    ax_client = AxClient()

    # Define the Search Space
    ax_client.create_experiment(
        name="poly_evolution_opt",
        parameters=[
            {
                "name": "population_size",
                "type": "choice",
                "value_type": "int",
                "values": [32, 64, 128, 256, 512, 1024, 2048], # Powers of two only
            },
            {
                "name": "mutation_rate",
                "type": "range",
                "bounds": [0.001, 0.4],
                "log_scale": True, # Log scale is better for learning rates/probabilities
            },
            {
                "name": "mutation_scale",
                "type": "range",
                "bounds": [1.0, 60.0],
            },
            {
                "name": "survival_rate",
                "type": "range",
                "bounds": [0.001, 0.5],
            },
            # Num_triangles changes the complexity, so it's fixed,
            # but can be optimized if needed:
            # { "name": "num_triangles", "type": "choice", "values": [50, 100] }
        ],
        objectives={"mse_loss": ObjectiveProperties(minimize=True)},
    )

    TOTAL_TRIALS = 200 # Number of experiments to run

    print(f"Starting {TOTAL_TRIALS} trials with Ax...")

    for i in range(TOTAL_TRIALS):
        parameters, trial_index = ax_client.get_next_trial()

        print(f"\n--- Trial {trial_index} ---")
        print(f"Params: {parameters}")

        # Run training with the suggested parameters
        try:
            val_loss = train_evaluate(parameters)
            print(f"Result Loss: {val_loss:.6f}")

            # Report results back to Ax
            ax_client.complete_trial(trial_index=trial_index, raw_data=val_loss)

        except Exception as e:
            print(f"Trial failed: {e}")
            ax_client.log_trial_failure(trial_index=trial_index)

    # --- RESULT ---
    best_parameters, values = ax_client.get_best_parameters()
    print("\n\n=== BEST PARAMETERS FOUND ===")
    print(best_parameters)
    print(f"Best Loss: {values[0]['mse_loss']}")

    # TODO: Save best config to YAML
    # save_best_config(best_parameters)

if __name__ == '__main__':
    main()
