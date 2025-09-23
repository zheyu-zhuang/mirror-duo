import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def log_SE3(T):
    """Log map from SE(3) to se(3): returns 6D vector (3 rot, 3 trans)"""
    R_mat = T[:3, :3]
    t = T[:3, 3]
    rotvec = R.from_matrix(R_mat).as_rotvec()
    return np.concatenate((rotvec, t))


def exp_SE3(xi):
    """Exp map from se(3) to SE(3): xi is 6D vector (3 rot, 3 trans)"""
    rotvec = xi[:3]
    t = xi[3:]
    R_mat = R.from_rotvec(rotvec).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T


def average_SE3(transformations, max_iters=10, tol=1e-6):
    """Averages a list of SE(3) matrices using Lie algebra"""
    T_mean = transformations[0].copy()  # Initialize with the first transformation
    for _ in range(max_iters):
        delta = np.zeros(6)
        for T in transformations:
            T_rel = np.linalg.inv(T_mean) @ T
            delta += log_SE3(T_rel)
        delta /= len(transformations)

        T_update = exp_SE3(delta)
        T_mean = T_mean @ T_update

        if np.linalg.norm(delta) < tol:
            break

    return T_mean


def SO3_from_6D(vec):
    """
    Recover an orthonormal SO(3) matrix from a 6-dimensional vector using the Gram-Schmidt process.

    Args:
        vec: A numpy array or torch tensor of shape (6,) or (N, 6) where the first 3 elements are the x axis and the next 3 elements are the y axis.

    Returns:
        A numpy array or torch tensor of shape (3, 3) or (N, 3, 3) representing the orthonormal SO(3) matrix.
    """
    if isinstance(vec, np.ndarray):
        backend = "numpy"
    elif isinstance(vec, torch.Tensor):
        backend = "torch"
    else:
        raise TypeError("Input must be a numpy array or a torch tensor")

    assert vec.shape[-1] == 6, "Input must be a 6D vector or a batch of 6D vectors"

    has_batch_dim = vec.ndim == 2
    vec = vec.reshape(-1, 6) if not has_batch_dim else vec

    x_axis = vec[:, :3]
    y_axis = vec[:, 3:]

    if backend == "numpy":
        # Normalize the x axis
        x_axis /= np.linalg.norm(x_axis, axis=-1, keepdims=True)

        # Make the y axis orthogonal to the x axis
        y_axis -= np.sum(x_axis * y_axis, axis=-1, keepdims=True) * x_axis
        y_axis /= np.linalg.norm(y_axis, axis=-1, keepdims=True)

        z_axis = np.cross(x_axis, y_axis)

        # Form the orthonormal SO(3) matrix
        SO3_matrix = np.stack([x_axis, y_axis, z_axis], axis=-1)

    elif backend == "torch":
        # Normalize the x axis
        x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)

        # Make the y axis orthogonal to the x axis
        y_axis = y_axis - torch.sum(x_axis * y_axis, dim=-1, keepdim=True) * x_axis
        y_axis = y_axis / torch.norm(y_axis, dim=-1, keepdim=True)

        z_axis = torch.cross(x_axis, y_axis, dim=-1)

        # Form the orthonormal SO(3) matrix
        SO3_matrix = torch.stack([x_axis, y_axis, z_axis], dim=-1)

    return SO3_matrix if has_batch_dim else SO3_matrix[0]


def SO3_to_6D(X):
    """
    Convert an orthonormal SO(3) matrix to a 6-dimensional vector.

    Args:
        X: A numpy array or torch tensor of shape (3, 3) or (N, 3, 3).

    Returns:
        A numpy array or torch tensor of shape (6,) or (N, 6).
    """
    assert isinstance(X, np.ndarray) or isinstance(
        X, torch.Tensor
    ), "Input must be a numpy array or a torch tensor"

    assert X.shape[-2:] == (3, 3), "Input must be a 3x3 matrix or a batch of 3x3 matrices"

    has_batch_dim = X.ndim == 3
    X = X.reshape(-1, 3, 3) if not has_batch_dim else X
    x_axis = X[:, :, 0]
    y_axis = X[:, :, 1]

    if isinstance(X, np.ndarray):
        vec_6D = np.concatenate([x_axis, y_axis], axis=-1)
    elif isinstance(X, torch.Tensor):
        vec_6D = torch.cat([x_axis, y_axis], dim=-1)

    return vec_6D if has_batch_dim else X[0]


def SO3_to_9D(X):
    """
    Convert an orthonormal SO(3) matrix to a 6-dimensional vector.

    Args:
        X: A numpy array or torch tensor of shape (3, 3) or (N, 3, 3).

    Returns:
        A numpy array or torch tensor of shape (9,) or (N, 9).
    """
    assert isinstance(X, np.ndarray) or isinstance(
        X, torch.Tensor
    ), "Input must be a numpy array or a torch tensor"

    assert X.shape[-2:] == (3, 3), "Input must be a 3x3 matrix or a batch of 3x3 matrices"
    has_batch_dim = X.ndim == 3
    X = X.reshape(-1, 3, 3) if not has_batch_dim else X
    return X.reshape(-1, 9) if has_batch_dim else X[0]


def SO3_from_9D(vec):
    """
    Convert a 9-dimensional vector to an orthonormal SO(3) matrix.

    Args:
        vec: A numpy array or torch tensor of shape (9,) or (N, 9).

    Returns:
        A numpy array or torch tensor of shape (3, 3) or (N, 3, 3).
    """
    assert isinstance(vec, np.ndarray) or isinstance(
        vec, torch.Tensor
    ), "Input must be a numpy array or a torch tensor"

    assert vec.shape[-1] == 9, "Input must be a 9D vector or a batch of 9D vectors"

    has_batch_dim = vec.ndim == 2
    vec = vec.reshape(-1, 9) if not has_batch_dim else vec

    rot = vec.reshape(-1, 3, 3)

    return rot if has_batch_dim else rot[0]


def SE3_to_6D_xyz(X: np.ndarray) -> np.ndarray:
    """
    Convert SE(3) 4x4 transformation matrices to 9D vectors.

    Args:
        X: A numpy array of shape (4, 4) or (N, 4, 4).

    Returns:
        A numpy array of shape (9,) or (N, 9).
    """
    assert isinstance(X, np.ndarray), "Input must be a numpy array"
    assert X.shape[-2:] == (4, 4), "Input must be a 4x4 matrix or a batch of 4x4 matrices"

    has_batch_dim = X.ndim == 3
    X = X.reshape(-1, 4, 4) if not has_batch_dim else X

    rot_6d = SO3_to_6D(X[:, :3, :3])
    t = X[:, :3, 3]
    vec_6D_xyz = np.concatenate([rot_6d, t], axis=-1)

    return vec_6D_xyz if has_batch_dim else vec_6D_xyz[0]


def SE3_to_xyz_6D(X: np.ndarray) -> np.ndarray:
    """
    Convert SE(3) 4x4 transformation matrices to 9D vectors.

    Args:
        X: A numpy array of shape (4, 4) or (N, 4, 4).

    Returns:
        A numpy array of shape (9,) or (N, 9).
    """
    assert isinstance(X, np.ndarray), "Input must be a numpy array"
    assert X.shape[-2:] == (4, 4), "Input must be a 4x4 matrix or a batch of 4x4 matrices"

    has_batch_dim = X.ndim == 3
    X = X.reshape(-1, 4, 4) if not has_batch_dim else X

    rot_6d = SO3_to_6D(X[:, :3, :3])
    t = X[:, :3, 3]
    vec_6D_xyz = np.concatenate([t, rot_6d], axis=-1)

    return vec_6D_xyz if has_batch_dim else vec_6D_xyz[0]


def SE3_from_6D_xyz(vec: np.ndarray) -> np.ndarray:
    """
    Convert 9D vectors to SE(3) 4x4 transformation matrices.

    Args:
        vec: A numpy array of shape (9,) or (N, 9).

    Returns:
        A numpy array of shape (4, 4) or (N, 4, 4).
    """
    assert isinstance(vec, np.ndarray), "Input must be a numpy array"
    assert vec.shape[-1] == 9, "Input must be a 9D vector or a batch of 9D vectors"

    has_batch_dim = vec.ndim == 2
    vec = vec.reshape(-1, 9) if not has_batch_dim else vec

    B = vec.shape[0]
    X = np.eye(4).reshape(1, 4, 4).repeat(B, axis=0)
    X[:, :3, :3] = SO3_from_6D(vec[:, :6])
    X[:, :3, 3] = vec[:, 6:]

    return X if has_batch_dim else X[0]


def SE3_from_xyz_6D(vec: np.ndarray) -> np.ndarray:
    """
    Convert 9D vectors to SE(3) 4x4 transformation matrices.

    Args:
        vec: A numpy array of shape (9,) or (N, 9).

    Returns:
        A numpy array of shape (4, 4) or (N, 4, 4).
    """
    assert isinstance(vec, np.ndarray), "Input must be a numpy array"
    assert vec.shape[-1] == 9, "Input must be a 9D vector or a batch of 9D vectors"

    has_batch_dim = vec.ndim == 2
    vec = vec.reshape(-1, 9) if not has_batch_dim else vec

    B = vec.shape[0]
    X = np.eye(4).reshape(1, 4, 4).repeat(B, axis=0)
    X[:, :3, :3] = SO3_from_6D(vec[:, 3:])
    X[:, :3, 3] = vec[:, :3]

    return X if has_batch_dim else X[0]


def SE3_to_6D_z(X: np.ndarray) -> np.ndarray:
    """
    Convert SE(3) 4x4 transformation matrices to 7D vectors.

    Args:
        X: A numpy array of shape (4, 4) or (N, 4, 4).

    Returns:
        A numpy array of shape (7,) or (N, 7).
    """
    assert isinstance(X, np.ndarray), "Input must be a numpy array"
    assert X.shape[-2:] == (4, 4), "Input must be a 4x4 matrix or a batch of 4x4 matrices"

    has_batch_dim = X.ndim == 3
    X = X.reshape(-1, 4, 4) if not has_batch_dim else X

    rot_6d = SO3_to_6D(X[:, :3, :3])
    t = X[:, 2, 3].reshape(-1, 1)
    vec_6D_z = np.concatenate([rot_6d, t], axis=-1)

    return vec_6D_z if has_batch_dim else vec_6D_z[0]


def SE3_to_9D_xyz(X: np.ndarray) -> np.ndarray:
    """
    Convert SE(3) 4x4 transformation matrices to 9D vectors.

    Args:
        X: A numpy array of shape (4, 4) or (N, 4, 4).

    Returns:
        A numpy array of shape (9,) or (N, 9).
    """
    assert isinstance(X, np.ndarray), "Input must be a numpy array"
    assert X.shape[-2:] == (4, 4), "Input must be a 4x4 matrix or a batch of 4x4 matrices"

    has_batch_dim = X.ndim == 3
    X = X.reshape(-1, 4, 4) if not has_batch_dim else X

    R = SO3_to_9D(X[:, :3, :3])
    t = X[:, :3, 3]
    vec_9D_xyz = np.concatenate([R, t], axis=-1)
    vec_9D_xyz = vec_9D_xyz.astype(np.float32)
    return vec_9D_xyz if has_batch_dim else vec_9D_xyz[0]


def SE3_from_9D_xyz(vec: np.ndarray) -> np.ndarray:
    """
    Convert 9D vectors to SE(3) 4x4 transformation matrices.

    Args:
        vec: A numpy array of shape (9,) or (N, 9).

    Returns:
        A numpy array of shape (4, 4) or (N, 4, 4).
    """
    assert isinstance(vec, np.ndarray), "Input must be a numpy array"
    assert vec.shape[-1] == 12, "Input must be a 12D vector or a batch of 12D vectors"

    has_batch_dim = vec.ndim == 2
    vec = vec.reshape(-1, 12) if not has_batch_dim else vec

    B = vec.shape[0]
    X = np.eye(4).reshape(1, 4, 4).repeat(B, axis=0)
    X[:, :3, :3] = SO3_from_9D(vec[:, :9])
    X[:, :3, 3] = vec[:, 9:]

    return X if has_batch_dim else X[0]


def SE3_to_9D_z(X: np.ndarray) -> np.ndarray:
    """
    Convert SE(3) 4x4 transformation matrices to 9D vectors.

    Args:
        X: A numpy array of shape (4, 4) or (N, 4, 4).

    Returns:
        A numpy array of shape (9,) or (N, 9).
    """
    assert isinstance(X, np.ndarray), "Input must be a numpy array"
    assert X.shape[-2:] == (4, 4), "Input must be a 4x4 matrix or a batch of 4x4 matrices"

    has_batch_dim = X.ndim == 3
    X = X.reshape(-1, 4, 4) if not has_batch_dim else X

    R = X[:, :3, :3].reshape(-1, 9)
    t = X[:, 2, 3].reshape(-1, 1)
    vec_9D_z = np.concatenate([R, t], axis=-1)
    vec_9D_z = vec_9D_z.astype(np.float32)
    return vec_9D_z if has_batch_dim else vec_9D_z[0]


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation

    print("Running tests...")
    B = 10000
    # Test SO3_to_6D
    rot = Rotation.random(B).as_matrix()
    rot6d = SO3_to_6D(rot)
    # Test SO3_from_6D
    rot_recovered = SO3_from_6D(rot6d)
    assert np.allclose(rot, rot_recovered), "SO3_to_6D and SO3_from_6D are not consistent"
    print("[Passed] SO3_to_6D and SO3_from_6D are consistent")
    # test torch and numpy consistency
    rot6d_torch = torch.tensor(rot6d)
    rot_recovered_torch = SO3_from_6D(rot6d_torch)
    assert torch.allclose(
        rot_recovered_torch, torch.tensor(rot_recovered)
    ), "SO3_from_6D is not consistent with numpy"
    print("[Passed] SO3_from_6D torch and numpy are consistent")
    # Test SE3_to_6D_xyz
    X = np.eye(4).reshape(1, 4, 4).repeat(B, axis=0)
    X[:, :3, :3] = rot
    X[:, :3, 3] = np.random.randn(B, 3)
    vec = SE3_to_6D_xyz(X)
    # Test SE3_from_6D_xyz
    X_recovered = SE3_from_6D_xyz(vec)
    assert np.allclose(X, X_recovered), "SE3_to_6D_xyz and SE3_from_6D_xyz are not consistent"
    print("[Passed] SE3_to_6D_xyz and SE3_from_6D_xyz are consistent")
    # Test SE3_to_9D_xyz
    vec = SE3_to_9D_xyz(X)
    # Test SE3_from_9D_xyz
    X_recovered = SE3_from_9D_xyz(vec)
    assert np.allclose(X, X_recovered), "SE3_to_9D_xyz and SE3_from_9D_xyz are not consistent"
    print("[Passed] SE3_to_9D_xyz and SE3_from_9D_xyz are consistent")
