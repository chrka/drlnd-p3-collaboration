import click
import numpy as np
import torch
from unityagents import UnityEnvironment

from agent import Agent

N_AGENTS = 2


def run(env, agent, n_episodes):
    """Run a pre-trained agent.

    Args:
        env (UnityEnvironment): Environment to run agent in
        agent (Agent): Agent to run
        n_episodes (int): Number of episodes to run
    """
    # Assume we're operating brain 0
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    mean_scores = []

    for i in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=False)[brain_name]

        states = env_info.vector_observations
        scores = np.zeros(N_AGENTS)

        step = 0
        while True:
            step += 1
            actions = np.array(
                [np.squeeze(agent.act(states[j], add_noise=False))
                 for j in range(N_AGENTS)])
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            states = next_states
            scores += rewards
            mean_score = np.mean(scores)
            print(f"\rStep: {step:3d}\tScore {mean_score}", end="")
            if np.any(dones):
                break

        mean_scores.append(mean_score)
        print(f"\nEpisode {i}\t Score {mean_score}")

    return mean_scores


@click.command()
@click.option('--environment', required=True,
              help="Path to Unity environment", type=click.Path())
@click.option('--n-episodes', default=3, help="Number of episodes to run")
@click.option('--weights-input', default='weights.pth', help="Network weights",
              type=click.Path())
def main(environment, n_episodes, weights_input):
    env = UnityEnvironment(file_name=environment)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = Agent(state_size=24, action_size=2, device=device)
    agent.load_weights(weights_input)

    run(env, agent, n_episodes)

    env.close()


if __name__ == '__main__':
    main()
