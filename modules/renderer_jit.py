import torch

@torch.jit.script
def render_loop_jit(genes: torch.Tensor,
                    grid_x: torch.Tensor,
                    grid_y: torch.Tensor,
                    width: int,
                    height: int) -> torch.Tensor:
    """
    JIT-compiled rendering loop for generating images from triangles.
    Runs purely on C++/CUDA level without Python interpreter overhead.

    Args:
        genes: Tensor containing triangle data. Shape: [Batch_Size, Num_Triangles, 10]
        grid_x: X-coordinate grid. Shape: [1, 1, Height, Width]
        grid_y: Y-coordinate grid. Shape: [1, 1, Height, Width]
        width: Image width
        height: Image height

    Returns:
        torch.Tensor: Rendered images batch. Shape: [Batch_Size, Height, Width, 3]
    """
    batch_size, num_triangles, _ = genes.shape
    device = genes.device

    # Initialize Canvas (White background)
    # Shape: [Batch_Size, Height, Width, 3]
    canvas = torch.ones((batch_size, height, width, 3), device=device, dtype=torch.float32)

    EPS = 1e-6

    # This loop is executed inside the CUDA graph / Fusion Kernel
    for i in range(num_triangles):
        tri = genes[:, i] # Shape: [Batch_Size, 10]

        # Unpacking and Reshaping for broadcasting
        # We reshape to [Batch, 1, 1, 1] to broadcast against the Grid [1, 1, H, W]
        # Using reshape() is preferred over view() for JIT reliability

        # Coordinates
        x1 = tri[:, 0:1].reshape(batch_size, 1, 1, 1)
        y1 = tri[:, 1:2].reshape(batch_size, 1, 1, 1)
        x2 = tri[:, 2:3].reshape(batch_size, 1, 1, 1)
        y2 = tri[:, 3:4].reshape(batch_size, 1, 1, 1)
        x3 = tri[:, 4:5].reshape(batch_size, 1, 1, 1)
        y3 = tri[:, 5:6].reshape(batch_size, 1, 1, 1)

        # Colors (Normalized to 0.0 - 1.0)
        r = tri[:, 6:7].reshape(batch_size, 1, 1, 1) * (1.0 / 255.0)
        g = tri[:, 7:8].reshape(batch_size, 1, 1, 1) * (1.0 / 255.0)
        b = tri[:, 8:9].reshape(batch_size, 1, 1, 1) * (1.0 / 255.0)
        alpha = tri[:, 9:10].reshape(batch_size, 1, 1, 1) * (1.0 / 255.0)

        # Concatenate RGB channels. Shape: [Batch, 1, 1, 3]
        color = torch.cat([r, g, b], dim=3)

        # --- Barycentric Coordinate Calculation ---

        # Denominator of the barycentric coordinate formulation
        denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)

        # Avoid division by zero
        denom = torch.where(torch.abs(denom) < EPS, torch.ones_like(denom) * EPS, denom)

        # Calculate weights w1, w2, w3 for every pixel
        # Broadcasting happens here: [Batch, 1, 1, 1] vs [1, 1, H, W] -> [Batch, 1, H, W]
        w1 = ((y2 - y3) * (grid_x - x3) + (x3 - x2) * (grid_y - y3)) / denom
        w2 = ((y3 - y1) * (grid_x - x3) + (x1 - x3) * (grid_y - y3)) / denom
        w3 = 1.0 - w1 - w2

        # Create Triangle Mask (pixel is inside if all weights >= 0)
        # Shape: [Batch, 1, Height, Width]
        mask = (w1 >= 0) & (w2 >= 0) & (w3 >= 0)

        # Blending Logic
        # Permute mask to match canvas channels: [Batch, H, W, 1]
        mask_f = mask.permute(0, 2, 3, 1).to(torch.float32)

        # Calculate alpha blending
        current_alpha = mask_f * alpha

        # Linear Interpolation: Canvas = Canvas * (1 - alpha) + Color * alpha
        canvas = canvas * (1.0 - current_alpha) + color * current_alpha

    return torch.clamp(canvas, 0.0, 1.0)

def render_batch(genes, width, height, grid_x, grid_y):
    """
    Wrapper function to call the JIT-compiled rendering loop.
    """
    return render_loop_jit(genes, grid_x, grid_y, width, height)
