from project3 import *
from tqdm import tqdm  # a convenient progress bar
import torch

N_EVALS = 500

if __name__ == "__main__":
    n_targets = 40
    agent = Agent()

    # This is a example of what the evaluation procedure looks like.
    # The whole dataset is divided into a training set and a test set.
    # The training set (including `data` and `label`) is distributed to you.
    # But in the final evaluation we will use the test set.

    data = torch.load("project3/Model_train/data.pth")
    label = data["label"]
    feature = data["feature"]

    scores = []
    
    for game in tqdm(range(N_EVALS)):
        # the class information is unavailable at test time.
        target_pos, target_features, target_cls, class_scores = generate_game(
            n_targets, N_CTPS, feature, label)
        ctps_inter = agent.get_action(
            target_pos, target_features, class_scores)
        score = evaluate(compute_traj(ctps_inter), target_pos,
                         class_scores[target_cls], RADIUS)
        scores.append(score)

    print(torch.stack(scores).float().mean())
