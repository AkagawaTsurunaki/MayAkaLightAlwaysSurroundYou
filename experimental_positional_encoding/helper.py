import torch
import matplotlib.pyplot as plt


def has_duplicate_rows(matrix):
    """
    To check if there are same row in the given matrix.
    :param matrix: torch.Tensor
    :return: True if given matrix has duplicate rows, False if not.
    """
    unique_rows = torch.unique(matrix, dim=1)
    return unique_rows.shape[1] < matrix.shape[1]


def draw(pe):
    """
    Draw the PE matrix as a plot.
    :param pe: torch.Tensor
    """
    plt.imshow(pe[0], cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar()
    plt.xlabel('max_len')
    plt.ylabel('d_model')
    plt.show()
