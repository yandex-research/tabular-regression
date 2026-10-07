# Source: https://github.com/jie040109/Conda/blob/df83c10717cacec9fa2a857fb1eb33837da8204c/conda_torch/projector.py
# fmt: off
# isort: off
import torch

class CondaProjector:
    def __init__(self, verbose=False, update_proj_gap=2000, scale=1.0, proj_type='std'):
        """
        Args:
            verbose (bool): Whether to print debug information.
            update_proj_gap (int): How often (in steps) to update the orthogonal matrix.
            scale (float): Scale factor to apply when projecting back.
            proj_type (str): Projection type ('std', 'reverse_std', 'right', 'left', or 'full').
        """
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type
        self.last_svd_step = -1  # Step at which SVD was last performed

    def project_with_cached_ortho(self, input_matrix, svd_basis_matrix, step):

        update_condition = self.ortho_matrix is None or step % self.update_proj_gap == 0
        already_updated_this_step = step == self.last_svd_step

        # Only update the orthogonal matrix if necessary and not already updated at this step
        if update_condition and not already_updated_this_step:
            if self.proj_type == 'std':
                if input_matrix.shape[0] >= input_matrix.shape[1]:
                    self.ortho_matrix = self.get_orthogonal_matrix(svd_basis_matrix, type='right')
                else:
                    self.ortho_matrix = self.get_orthogonal_matrix(svd_basis_matrix, type='left')
            elif self.proj_type == 'reverse_std':
                if input_matrix.shape[0] >= input_matrix.shape[1]:
                    self.ortho_matrix = self.get_orthogonal_matrix(svd_basis_matrix, type='left')
                else:
                    self.ortho_matrix = self.get_orthogonal_matrix(svd_basis_matrix, type='right')
            elif self.proj_type == 'right':
                self.ortho_matrix = self.get_orthogonal_matrix(svd_basis_matrix, type='right')
            elif self.proj_type == 'left':
                self.ortho_matrix = self.get_orthogonal_matrix(svd_basis_matrix, type='left')
            elif self.proj_type == 'full':
                self.ortho_matrix = self.get_orthogonal_matrix(svd_basis_matrix, type='full')
            else:
                raise ValueError(f"Unknown proj_type: {self.proj_type}")
            self.last_svd_step = step

        return self.project(input_matrix, svd_basis_matrix, step, cached_ortho_matrix=self.ortho_matrix)

    def project(self, input_matrix, svd_basis_matrix, step, cached_ortho_matrix=None):

        if cached_ortho_matrix is not None:
            self.ortho_matrix = cached_ortho_matrix

        device = input_matrix.device
        ortho = self.ortho_matrix

        if self.proj_type == 'std':
            if input_matrix.shape[0] >= input_matrix.shape[1]:
                projected_matrix = torch.matmul(input_matrix, ortho.t().to(device))
            else:
                projected_matrix = torch.matmul(ortho.t().to(device), input_matrix)
        elif self.proj_type == 'reverse_std':
            if input_matrix.shape[0] >= input_matrix.shape[1]:
                projected_matrix = torch.matmul(ortho.t().to(device), input_matrix)
            else:
                projected_matrix = torch.matmul(input_matrix, ortho.t().to(device))
        elif self.proj_type == 'right':
            projected_matrix = torch.matmul(input_matrix, ortho.t().to(device))
        elif self.proj_type == 'left':
            projected_matrix = torch.matmul(ortho.t().to(device), input_matrix)
        elif self.proj_type == 'full':
            # ortho is a list of [U, Vh]
            left, right = ortho
            projected_matrix = torch.matmul(left.t().to(device), input_matrix)
            projected_matrix = torch.matmul(projected_matrix, right.t().to(device))
        else:
            raise ValueError(f"Unknown proj_type: {self.proj_type}")

        return projected_matrix

    def project_back(self, projected_matrix):

        device = projected_matrix.device
        ortho = self.ortho_matrix

        if self.proj_type == 'std':
            if projected_matrix.shape[0] >= projected_matrix.shape[1]:
                projected_back_matrix = torch.matmul(projected_matrix, ortho.to(device))
            else:
                projected_back_matrix = torch.matmul(ortho.to(device), projected_matrix)
        elif self.proj_type == 'reverse_std':
            if projected_matrix.shape[0] <= projected_matrix.shape[1]:
                projected_back_matrix = torch.matmul(ortho.to(device), projected_matrix)
            else:
                projected_back_matrix = torch.matmul(projected_matrix, ortho.to(device))
        elif self.proj_type == 'right':
            projected_back_matrix = torch.matmul(projected_matrix, ortho.to(device))
        elif self.proj_type == 'left':
            projected_back_matrix = torch.matmul(ortho.to(device), projected_matrix)
        elif self.proj_type == 'full':
            # ortho is a list of [U, Vh]
            left, right = ortho
            projected_back_matrix = torch.matmul(left.to(device), projected_matrix)
            projected_back_matrix = torch.matmul(projected_back_matrix, right.to(device))
        else:
            raise ValueError(f"Unknown proj_type: {self.proj_type}")

        return projected_back_matrix * self.scale

    def get_orthogonal_matrix(self, svd_basis_matrix, type):
        """
        Compute the orthogonal matrix (U or Vh, or both) via SVD.

        Args:
            svd_basis_matrix (Tensor): Input matrix for SVD.
            type (str): 'left', 'right', or 'full'.

        Returns:
            Tensor or list: U, Vh, or [U, Vh] from SVD.
        """
        matrix = svd_basis_matrix.data
        orig_dtype = matrix.dtype
        orig_device = matrix.device

        # Perform SVD in float32 for numerical stability
        if orig_dtype != torch.float:
            matrix = matrix.float()

        U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)

        if type == 'right':
            B = Vh
            if orig_dtype != torch.float:
                B = B.to(orig_device).type(orig_dtype)
            return B
        elif type == 'left':
            A = U
            if orig_dtype != torch.float:
                A = A.to(orig_device).type(orig_dtype)
            return A
        elif type == 'full':
            A = U
            B = Vh
            if orig_dtype != torch.float:
                A = A.to(orig_device).type(orig_dtype)
                B = B.to(orig_device).type(orig_dtype)
            return [A, B]
        else:
            raise ValueError("type should be 'left', 'right', or 'full'")