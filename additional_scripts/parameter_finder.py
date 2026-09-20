from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np

from agent.brain_network import BrainNetwork


def get_newest_folder(logs_dir: Path) -> Path:
    """Finds the most recently modified directory in the logs directory."""
    folders = [p for p in logs_dir.iterdir() if p.is_dir()]
    if not folders:
        raise FileNotFoundError(f"No subdirectories found in {logs_dir}")
    return max(folders, key=lambda p: p.stat().st_mtime)


def parse_continuation(stats_file: Path) -> Path | None:
    """Extracts the session continuation path from execution_statistics.txt."""
    if not stats_file.exists():
        return None

    pattern = re.compile(r"CONTINUATION FROM SESSION:\s*(.+)", re.IGNORECASE)
    with stats_file.open("r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line.strip())
            if match:
                raw_path = match.group(1).strip()
                return Path(raw_path)
    return None


def collect_continuation_chain(start_dir: Path, logs_root: Path) -> list[Path]:
    """
    Traverses backward through execution_statistics.txt continuation chains,
    returning an ordered list of session directories from oldest to newest.
    """
    chain: list[Path] = [start_dir]
    visited = {start_dir.resolve()}
    current_dir = start_dir

    while True:
        stats_file = current_dir / "execution_statistics.txt"
        continued_rel_path = parse_continuation(stats_file)

        if not continued_rel_path:
            break

        if (logs_root.parent / continued_rel_path).exists():
            next_dir = logs_root.parent / continued_rel_path
        elif (logs_root / continued_rel_path.name).exists():
            next_dir = logs_root / continued_rel_path.name
        else:
            print(f"[Chain] Warning: Continuation folder '{continued_rel_path}' not found. Stopping traversal.")
            break

        next_resolved = next_dir.resolve()
        if next_resolved in visited:
            print(f"[Chain] Loop detected at {next_dir}. Stopping traversal.")
            break

        visited.add(next_resolved)
        chain.append(next_dir)
        current_dir = next_dir

    # Reverse chain so that oldest sessions appear first in chronological order
    chain.reverse()
    return chain


def load_chained_dataset(session_dirs: list[Path], split_ratio: float = 0.75):
    """
    Loads and aggregates offline_transitions.npz across a chain of session folders,
    then splits the combined data into train and validation sets.
    """
    all_prev_states = []
    all_skill_names = []
    all_post_states = []
    all_succeeds = []

    for s_dir in session_dirs:
        dataset_path = s_dir / "offline_transitions.npz"
        if not dataset_path.exists():
            print(f"[Dataset] Skipping '{s_dir.name}': no 'offline_transitions.npz' found.")
            continue

        print(f"[Dataset] Loading transitions from: {s_dir.name}")
        data = np.load(dataset_path, allow_pickle=True)
        all_prev_states.append(data["prev_states"])
        all_skill_names.append(data["skill_names"])
        all_post_states.append(data["post_states"])
        all_succeeds.append(data["succeeds"])

    if not all_succeeds:
        raise FileNotFoundError("No 'offline_transitions.npz' files found across the continuation chain.")

    prev_states = np.concatenate(all_prev_states, axis=0)
    skill_names = np.concatenate(all_skill_names, axis=0)
    post_states = np.concatenate(all_post_states, axis=0)
    succeeds = np.concatenate(all_succeeds, axis=0)

    n_samples = len(succeeds)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(split_ratio * n_samples)
    train_idx, val_idx = indices[:split], indices[split:]

    train_data = [
        (prev_states[i], str(skill_names[i]), post_states[i], bool(succeeds[i]))
        for i in train_idx
    ]
    val_data = [
        (prev_states[i], str(skill_names[i]), post_states[i], bool(succeeds[i]))
        for i in val_idx
    ]

    return train_data, val_data


def evaluate_brain(brain: BrainNetwork, val_data: list):
    """Computes F1-score across preconditions and transition edge density."""
    tp, fp, fn, tn = 0, 0, 0, 0

    for prev_state, skill_name, _, actual_success in val_data:
        predicted_met = brain.preconditions_met(skill_name, prev_state)
        if predicted_met and actual_success:
            tp += 1
        elif predicted_met and not actual_success:
            fp += 1
        elif not predicted_met and actual_success:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    total_skills = len(brain.trans_graph.nodes)
    if total_skills <= 1:
        edge_connectivity = 0.0
    else:
        total_edges = sum(len(sources) for sources in brain.trans_graph.edges_by_target.values())
        max_possible_edges = total_skills * (total_skills - 1)
        edge_connectivity = min(1.0, total_edges / max(1, max_possible_edges))

    return f1, edge_connectivity


def sample_hyperparameters() -> dict:
    """Generates a random parameter configuration within bounded intervals."""
    return {
        "max_points": random.randint(20, 100),
        "base_radius": float(np.exp(random.uniform(np.log(0.05), np.log(0.40)))),
        "max_radius": random.uniform(0.40, 1.80),
        "learning_rate_b": random.uniform(0.02, 0.20),
        "learning_rate_n": random.uniform(0.001, 0.02),
        "max_edge_age": random.randint(20, 100),
        "lambda_step": random.randint(2, 20),
        "alpha_threshold": random.uniform(0.4, 0.95),
        "d": random.uniform(0.985, 0.999),
    }


def run_trial(params: dict, train_data: list, val_data: list):
    """Initializes, trains, and scores a brain instance on transition data."""
    brain = BrainNetwork(log_dir=None)
    brain.max_points = params["max_points"]
    brain.base_radius = params["base_radius"]
    brain.max_radius = params["max_radius"]
    brain.learning_rate_b = params["learning_rate_b"]
    brain.learning_rate_n = params["learning_rate_n"]
    brain.max_edge_age = params["max_edge_age"]
    brain.lambda_step = params["lambda_step"]
    brain.alpha_threshold = params["alpha_threshold"]
    brain.d = params["d"]

    batch_size = 10
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i : i + batch_size]
        brain.update(batch)

    f1, connectivity = evaluate_brain(brain, val_data)
    composite_score = (0.75 * f1) + (0.25 * connectivity)
    return composite_score, f1, connectivity


def main():
    parser = argparse.ArgumentParser(description="Optimize BrainNetwork GNG parameters over session continuation chains.")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Root directory containing session logs")
    parser.add_argument("--session", type=str, default=None, help="Head session folder name (defaults to newest)")
    parser.add_argument("--iterations", type=int, default=50, help="Number of random search iterations")
    args = parser.parse_args()

    logs_root = Path(args.logs_dir)
    if not logs_root.exists():
        raise FileNotFoundError(f"Logs root '{logs_root}' does not exist.")

    if args.session:
        target_dir = logs_root / args.session
        if not target_dir.exists():
            raise FileNotFoundError(f"Session directory '{target_dir}' does not exist.")
    else:
        target_dir = get_newest_folder(logs_root)

    print(f"Head session directory: {target_dir.name}")
    chain = collect_continuation_chain(target_dir, logs_root)
    print(f"Discovered continuation chain ({len(chain)} sessions): {' -> '.join(p.name for p in chain)}")

    train_data, val_data = load_chained_dataset(chain)
    print(f"Aggregated dataset: {len(train_data)} train samples, {len(val_data)} validation samples.\n")

    best_score = -1.0
    best_params = None
    best_metrics = (0.0, 0.0)

    for it in range(1, args.iterations + 1):
        params = sample_hyperparameters()
        score, f1, conn = run_trial(params, train_data, val_data)

        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = (f1, conn)
            print(f"[*] Iteration {it:03d} -> NEW BEST: Score={score:.4f} (F1={f1:.4f}, Conn={conn:.4f})")
        else:
            print(f"    Iteration {it:03d} -> Score={score:.4f} (F1={f1:.4f}, Conn={conn:.4f})")

    print("\n" + "=" * 60)
    print("OPTIMIZATION FINISHED")
    print(f"Best Composite Score: {best_score:.4f}")
    print(f"Best F1-Score:        {best_metrics[0]:.4f}")
    print(f"Best Connectivity:    {best_metrics[1]:.4f}")
    print("-" * 60)
    print("Assign these inside your BrainNetwork.__init__:")
    if best_params:
        for key, val in best_params.items():
            if isinstance(val, float):
                print(f"self.{key} = {val:.4f}")
            else:
                print(f"self.{key} = {val}")
    print("=" * 60)


if __name__ == "__main__":
    main()