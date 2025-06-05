import subprocess
import random

def random_choice_with_prob(options, prob):
    if random.random() < prob:
        return random.choice(options)
    return None

def sample_params():
    learning_rate = random.uniform(1e-5, 1e-2)
    optimizer = random.choice(["adam", "adamw", "sgd"])
    augment = random_choice_with_prob([True], prob=0.5)
    activation = random.choice(["relu", "relu6", "tanh", "sigmoid", "leaky_relu"])
    dropout = round(random.uniform(0.0, 0.2), 2)
    return learning_rate, optimizer, augment, activation, dropout

def build_command(params, trial_id):
    lr, opt, aug, act, drop = params
    wandb_name = f"random_search_run_{trial_id}"
    cmd = [
        "CUDA_VISIBLE_DEVICES=0",
        "python3", "train.py",
        "--epochs", "100",
        "--use_wandb",
        "--wandb_name", wandb_name,
        "--data_dir", "dataset/filtered_dataset.pkl",
        "--early_stopping_patience", "10",
        "--num_workers", "2",
        "--batch_size", "64",
        "--learning_rate", f"{lr:.6f}",
        "--optimizer", opt,
        "--activation", act,
        "--dropout", str(drop)
    ]
    if aug:
        cmd.append("--augment")

    return " ".join(cmd)

def run_trial(trial_id):
    params = sample_params()
    cmd = build_command(params, trial_id)
    print(f"\nTrial {trial_id}: Running command:\n{cmd}")

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Trial {trial_id} FAILED with return code {result.returncode}")

def main(num_trials=10):
    for i in range(1, num_trials+1):
        run_trial(i)

if __name__ == "__main__":
    main(num_trials=10)
