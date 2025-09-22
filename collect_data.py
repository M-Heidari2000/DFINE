import argparse
from dfine.env_utils import make_env
from minari import DataCollector
from tqdm import tqdm
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.ppo import PPO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="collect data from environments")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="gym environment to collect data from")
    parser.add_argument("--num-ppo-steps", type=int, default=10000, help="number of ppo training steps")
    parser.add_argument("--num-episodes", type=int, default=100, help="number of episodes to collect")
    parser.add_argument("--dataset-id", type=str, required=True, help="the name associated to the collected dataset")
    parser.add_argument("--author-name", type=str, required=True, help="author associated with the collected dataset")
    parser.add_argument("--action-repeat", type=int, default=2, help="action repeat")

    args = parser.parse_args()

    orig_env = make_env(id=args.env, action_repeat=args.action_repeat)
    check_env(env=orig_env)

    if args.num_ppo_steps > 0:
        model = PPO(policy="MlpPolicy", env=orig_env, device="cpu", verbose=True)
        model.learn(total_timesteps=args.num_ppo_steps)

    env = DataCollector(env=orig_env, record_infos=True)
    for _ in tqdm(range(args.num_episodes)):
        obs, info = env.reset()
        done = False
        while not done:
            if args.num_ppo_steps > 0:
                action, _ = model.predict(observation=obs)
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action=action)
            done = terminated or truncated

    dataset = env.create_dataset(
        dataset_id=args.dataset_id,
        eval_env=orig_env,
        algorithm_name=f"ppo-{args.num_ppo_steps}steps" if args.num_ppo_steps > 0 else "random",
        author=args.author_name,
        description=f"action repeat: {args.action_repeat}",
    )