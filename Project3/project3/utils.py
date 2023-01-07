from project3.src import generate_game, N_CTPS, evaluate, compute_traj, RADIUS, torch
import sys
import time

INT_MAX = 2147483647


def choose_ctps(target_pos, target_score):
    start_time = time.time()

    max_score = -INT_MAX
    max_ctps = None
    while time.time() - start_time < 0.25:
        ctps = torch.rand(
            (N_CTPS-2, 2)) * torch.tensor([N_CTPS-2, 2.]) + torch.tensor([1., -1.])
        

        score = evaluate(compute_traj(
            ctps), target_pos, target_score, RADIUS)
        if max_score < score:
            max_score = score
            max_ctps = ctps
    return max_ctps
