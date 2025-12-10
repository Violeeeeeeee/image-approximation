import torch
import numpy as np
import random
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from deap import base, creator, tools

from modules.renderer_jit import render_batch

class EvoPipeline:
    """
    Manages the evolutionary process using DEAP and PyTorch.
    Handles data loading, population initialization, and the main evolution loop.
    """
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

        # Set random seeds for reproducibility
        seed = config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self._setup_data()
        self._setup_deap()

    def _setup_data(self):
        """
        Loads the target image, resizes it, and prepares tensors for the GPU.
        Initializes the coordinate grid used for JIT rendering.
        """
        target_path = self.cfg['data']['target_path']
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Image not found: {target_path}")

        img = cv2.imread(target_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        resize_to = self.cfg['data'].get('resize_to', 128)
        if resize_to:
            img = cv2.resize(img, (resize_to, resize_to))

        self.h, self.w, _ = img.shape
        print(f"Target Image: {self.w}x{self.h} on {self.device}")

        # Prepare target tensor.
        # Shape: [1, Height, Width, 3] (Channels Last)
        self.target_tensor = torch.from_numpy(img).float().to(self.device) / 255.0
        self.target_tensor = self.target_tensor.unsqueeze(0)

        # Cache coordinate grids for JIT renderer to avoid re-creation every batch.
        y_coords = torch.linspace(0, self.h - 1, self.h, device=self.device)
        x_coords = torch.linspace(0, self.w - 1, self.w, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Reshape grids for broadcasting during rendering.
        # Shape: [1, 1, Height, Width]
        self.grid_x = grid_x.unsqueeze(0).unsqueeze(0)
        self.grid_y = grid_y.unsqueeze(0).unsqueeze(0)

    def _create_individual_tensor(self):
        """
        Generates a random genome (PyTorch Tensor) for a single individual.

        Returns:
            torch.Tensor: A tensor representing N triangles.
            Shape: [Num_Triangles, 10] where 10 = (x1,y1, x2,y2, x3,y3, r,g,b, a)
        """
        n_tri = self.cfg['algorithm']['num_triangles']
        genes = torch.zeros((n_tri, 10), device=self.device)

        # Coordinates [x1, y1, x2, y2, x3, y3]
        # Range: [0, Width] for X, [0, Height] for Y
        genes[:, 0:6:2] = torch.rand(n_tri, 3, device=self.device) * self.w
        genes[:, 1:6:2] = torch.rand(n_tri, 3, device=self.device) * self.h

        # RGB Colors
        # Range: [0, 255]
        genes[:, 6:9]   = torch.rand(n_tri, 3, device=self.device) * 255

        # Alpha (Transparency)
        # Range: [30, 100] approximately
        genes[:, 9]     = torch.rand(n_tri, device=self.device) * 70 + 30
        return genes

    def _setup_deap(self):
        """
        Initializes DEAP toolbox, creator, and registers evolutionary operators.
        """
        # FitnessMin: We minimize loss (-1.0 weight)
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        # Registration of the generator
        self.toolbox.register("attr_tensor", self._create_individual_tensor)

        # Individual is a list containing one tensor element: [Tensor(N, 10)]
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_tensor, n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Registration of operators
        self.toolbox.register("evaluate_batch", self.evaluate_batch_gpu)
        self.toolbox.register("mate", self.cx_uniform_tensor)
        self.toolbox.register("mutate", self.mut_gaussian_tensor)
        self.toolbox.register("select", tools.selBest) # Elitism selection

    # --- GPU OPERATORS ---

    def cx_uniform_tensor(self, ind1, ind2):
        """
        Custom Uniform Crossover for tensors.
        Randomly swaps genes (triangles) between two individuals based on a mask.

        Args:
            ind1, ind2: Individuals containing gene tensors of shape [Num_Triangles, 10].
        """
        # ind1[0] is the tensor of shape [N_Tri, 10]
        mask = (torch.rand_like(ind1[0]) > 0.5).float()

        # Mix genes based on the mask
        new_1 = ind1[0] * mask + ind2[0] * (1 - mask)
        new_2 = ind2[0] * mask + ind1[0] * (1 - mask)

        ind1[0] = new_1
        ind2[0] = new_2
        return ind1, ind2

    def mut_gaussian_tensor(self, ind):
        """
        Custom Gaussian Mutation for tensors.
        Adds random noise to gene values with specific probability.

        Args:
            ind: Individual containing gene tensor of shape [Num_Triangles, 10].
        """
        rate = self.cfg['algorithm']['mutation_rate']
        scale = self.cfg['algorithm']['mutation_scale']

        tensor = ind[0] # Shape: [N_Tri, 10]

        # Create a boolean mask for genes to mutate
        mask = (torch.rand_like(tensor) < rate).float()

        # Generate Gaussian noise
        noise = torch.randn_like(tensor) * scale

        # Apply mutation
        tensor = tensor + mask * noise

        # Clamp values to valid ranges (Coordinates within image, Color 0-255)
        tensor[:, 0:6:2] = tensor[:, 0:6:2].clamp(0, self.w) # X coords
        tensor[:, 1:6:2] = tensor[:, 1:6:2].clamp(0, self.h) # Y coords
        tensor[:, 6:10]  = tensor[:, 6:10].clamp(0, 255)     # RGBA

        ind[0] = tensor
        return ind,

    def evaluate_batch_gpu(self, population):
        """
        Evaluates the fitness of the entire population in a single batch on the GPU.

        Key optimization: Stacks individual tensors into a batch tensor,
        renders all images simultaneously, and computes MSE loss.

        Args:
            population: List of individuals.

        Returns:
            numpy.ndarray: Array of MSE scores.
        """
        # Stack all individuals into a single batch tensor
        # Shape: [Pop_Size, Num_Triangles, 10]
        batch_genes = torch.stack([ind[0] for ind in population])

        # Render images using the JIT compiled renderer
        # Output Shape: [Pop_Size, Height, Width, 3]
        rendered_images = render_batch(batch_genes, self.w, self.h, self.grid_x, self.grid_y)

        # Calculate Mean Squared Error (MSE)
        # diff Shape: [Pop_Size, Height, Width, 3]
        diff = (rendered_images - self.target_tensor) ** 2

        # Mean over spatial dimensions and channels -> [Pop_Size]
        mse_scores = diff.mean(dim=[1, 2, 3])

        return mse_scores.cpu().numpy()

    def run(self):
        """
        Executes the main evolutionary loop.
        """
        print("--- Starting DEAP + Torch Evolution ---")
        pop_size = self.cfg['algorithm']['population_size']
        generations = self.cfg['algorithm']['generations']
        survival_rate = self.cfg['algorithm']['survival_rate']

        out_path = self.cfg['data']['output_path']
        os.makedirs(out_path, exist_ok=True)

        # 1. Initialize Population
        pop = self.toolbox.population(n=pop_size)
        loss_history = []

        pbar = tqdm(range(generations))

        # Calculate elitism counts
        n_elites = int(pop_size * survival_rate)
        n_offspring = pop_size - n_elites

        for gen in pbar:
            # --- EVALUATION ---
            # Evaluate all individuals at once (Batch Processing)
            fitnesses = self.toolbox.evaluate_batch(pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = (fit,)

            # Stats tracking
            current_best_loss = np.min(fitnesses)
            loss_history.append(current_best_loss)
            pbar.set_description(f"Loss: {current_best_loss:.6f}")

            # --- SELECTION (Elitism) ---
            # Select the best individuals to survive without mutation
            elites = self.toolbox.select(pop, n_elites)

            # --- REPRODUCTION ---
            offspring = []
            while len(offspring) < n_offspring:
                # Randomly select parents from the elite group
                parent1 = random.choice(elites)
                parent2 = random.choice(elites)

                # Clone parents (Deep Copy of the tensor is crucial)
                child1 = creator.Individual([parent1[0].clone()])
                child2 = creator.Individual([parent2[0].clone()])

                # Crossover
                self.toolbox.mate(child1, child2)

                # Mutation
                self.toolbox.mutate(child1)
                self.toolbox.mutate(child2)

                offspring.append(child1)
                if len(offspring) < n_offspring:
                    offspring.append(child2)

            # Create new population: Elites (cloned to preserve archives) + Offspring
            next_gen_elites = [creator.Individual([e[0].clone()]) for e in elites]
            pop = next_gen_elites + offspring

            # --- CHECKPOINTING ---
            if gen % 500 == 0 or gen == generations - 1:
                self.save_checkpoint(elites[0], gen, out_path)

        # Final Visualization
        plt.figure()
        plt.plot(loss_history)
        plt.title("Evolution Progress")
        plt.grid()
        plt.xlabel("Generation")
        plt.ylabel("MSE Loss")
        plt.savefig(os.path.join(out_path, "loss_plot.png"))
        print("Evolution Finished.")

    def save_checkpoint(self, best_ind, gen, path):
        """
        Saves the current best image and its gene tensor to disk.
        """
        with torch.no_grad():
            genes = best_ind[0].unsqueeze(0) # Add batch dim: [1, N, 10]
            img_tensor = render_batch(genes, self.w, self.h, self.grid_x, self.grid_y)

            # Convert to Numpy uint8 format
            img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(path, f"gen_{gen}.png"), img_bgr)
            torch.save(genes, os.path.join(path, "best_genes.pt"))
