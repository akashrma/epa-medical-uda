import json
import os
import math

import numpy as np
import torch


def _find_repo_root(start_dir):
    cur = os.path.abspath(start_dir)
    for _ in range(6):
        if os.path.isfile(os.path.join(cur, "README.md")) and os.path.isfile(
                os.path.join(cur, "AGENTS.md")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(start_dir)


def get_default_etf_dir():
    repo_root = _find_repo_root(os.path.dirname(__file__))
    return os.path.join(repo_root, "etf_prototypes")


def _prototype_paths(save_dir, num_classes, feature_dim):
    base_name = "etf_prototypes_K{}_D{}".format(num_classes, feature_dim)
    pt_path = os.path.join(save_dir, base_name + ".pt")
    npy_path = os.path.join(save_dir, base_name + ".npy")
    meta_path = os.path.join(save_dir, base_name + ".json")
    return pt_path, npy_path, meta_path


def generate_random_orthogonal_matrix(feature_dim, num_classes):
    a = np.random.random(size=(feature_dim, num_classes))
    p, _ = np.linalg.qr(a)
    p = torch.tensor(p, dtype=torch.float32)
    if not torch.allclose(p.t() @ p, torch.eye(num_classes), atol=1e-7):
        raise ValueError("Orthogonality issue!")
    return p


def load_etf_class_prototypes(feature_dim,
                              num_classes,
                              device="cuda",
                              save_dir=None):
    if save_dir is None:
        save_dir = get_default_etf_dir()
    pt_path, _, meta_path = _prototype_paths(save_dir, num_classes,
                                             feature_dim)
    if not (os.path.exists(pt_path) and os.path.exists(meta_path)):
        return None
    try:
        m_star = torch.load(pt_path, map_location=device)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if m_star.shape != (num_classes, feature_dim):
            return None
        if meta.get("num_classes") != num_classes or meta.get(
                "feature_dim") != feature_dim:
            return None
        return m_star.to(device)
    except Exception:
        return None


def generate_etf_class_prototypes(feature_dim,
                                  num_classes,
                                  device="cuda",
                                  save_dir=None):
    """
    Generate ETF class prototypes with metadata and dual-format saving.
    File: etf_prototypes_K{num_classes}_D{feature_dim}.pt/.npy/.json
    """
    if save_dir is None:
        save_dir = get_default_etf_dir()
    os.makedirs(save_dir, exist_ok=True)
    pt_path, npy_path, meta_path = _prototype_paths(save_dir, num_classes,
                                                    feature_dim)

    p = generate_random_orthogonal_matrix(feature_dim, num_classes)
    eye = torch.eye(num_classes)
    ones = torch.ones(num_classes, num_classes)
    scale = math.sqrt(num_classes / (num_classes - 1))
    m_star = scale * p @ (eye - (1.0 / num_classes) * ones)
    m_star = m_star.t().contiguous().to(device)

    torch.save(m_star, pt_path)
    np.save(npy_path, m_star.cpu().numpy())
    with open(meta_path, "w") as f:
        json.dump(
            {
                "num_classes": num_classes,
                "feature_dim": feature_dim,
            },
            f,
        )

    return m_star
