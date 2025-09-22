import os
import json
import wandb
import torch
import minari
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dfine.memory import ReplayBuffer
from dfine.train import train_backbone, train_cost
from dfine.test import test


def generate_id():
    """
        generates run id based on the time
    """
    return datetime.now().strftime("%Y%m%d_%H%M")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DFINE")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--action-repeat", type=int, default=2, help="action repeat")
    parser.add_argument("--log-dir", type=str, default="log", help="logging directory")
    parser.add_argument("--run-id", type=str, default=generate_id(), help="id associated with this run")
    parser.add_argument("--dataset", type=str, default="classic-control/pendulum/medium-v0", help="name of the minari dataset")
    parser.add_argument("--num-updates", type=int, default=2500, help="number of gradient descent steps")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="train-test split ratio")
    parser.add_argument("--test-interval", type=int, default=10, help="number of training steps before testing")
    parser.add_argument("--x-dim", type=int, default=30, help="x(state) dimension")
    parser.add_argument("--a-dim", type=int, default=100, help="a(intermediate state) dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="hidden layer dimension for encoder and decoder")
    parser.add_argument("--min-var", type=float, default=0.01, help="minimum var for states")
    parser.add_argument("--dropout-p", type=float, default=0.4, help="dropout ratio for encoder and decoder")
    parser.add_argument("--chunk-length", type=int, default=50, help="length of chunks used for the update step")
    parser.add_argument("--prediction-k", type=int, default=24, help="number of future steps prediction")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--eps", type=float, default=1e-8, help="epsilon for optimizer")
    parser.add_argument("--clip-grad-norm", type=float, default=1000.0, help="clip gradients to this value")
    parser.add_argument("--disable-gpu", action="store_true", default=False, help="disable using gpu")
    parser.add_argument("--num-test-episodes", type=int, default=10, help="number of test episodes")
    parser.add_argument("--planning-horizon", type=int, default=12, help="planning horizon for iLQR")
    parser.add_argument("--action-noise-std", type=float, default=0.3, help="action noise for exploration")

    args = parser.parse_args()

    # prepare logging
    save_dir = Path(args.log_dir) / args.run_id
    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f)
    
    wandb.init(
        project="Controlling from high-dimensional observations",
        name="DFINE",
        config=vars(args),
    )

    wandb.define_metric("global_step")
    wandb.define_metric("*",step_metric="global_step")

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # load the dataset
    dataset = minari.load_dataset(args.dataset)
    test_size = int(len(dataset) * args.test_ratio)
    train_size = len(dataset) - test_size
    train_data, test_data = minari.split_dataset(dataset=dataset, sizes=[train_size, test_size])
    train_buffer = ReplayBuffer.load_from_minari(dataset=train_data)
    test_buffer = ReplayBuffer.load_from_minari(dataset=test_data)

    print("training backbone ...")
    encoder, decoder, dynamics_model = train_backbone(
        args=args,
        train_buffer=train_buffer,
        test_buffer=test_buffer,
    )
    
    print("training cost model ...")
    cost_model = train_cost(
        args=args,
        encoder=encoder,
        decoder=decoder,
        dynamics_model=dynamics_model,
        train_buffer=train_buffer,
        test_buffer=test_buffer,
    )

    print("testing ...")
    env = dataset.recover_environment()
    test(
        args=args,
        env=env,
        encoder=encoder,
        dynamics_model=dynamics_model,
        cost_model=cost_model
    )

    wandb.finish()