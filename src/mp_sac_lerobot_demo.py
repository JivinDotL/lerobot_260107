import multiprocessing as mp
from pathlib import Path
import signal
import gymnasium as gym
import torch
import time

from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.rl.buffer import ReplayBuffer
import gym_hil

# ===============================================================
# 1. Actor process
# ===============================================================
def run_actor(transitions_q, params_q, stop_event, policy_actor, env_name, device):

    print("[Actor] Started")
    env = gym.make(env_name, render_mode="rgb_array", image_obs=False)

    obs, _ = env.reset()

    while not stop_event.is_set():

        # 接收来自 learner 的参数
        try:
            new_params = params_q.get_nowait()
            policy_actor.load_state_dict(new_params)
            # print("[Actor] Updated model params")
        except:
            pass

        with torch.no_grad():
            obs_tensor = {k: torch.tensor(v).unsqueeze(0).float().to(device)
                          for k, v in obs.items()}
            action = policy_actor.act(obs_tensor)["action"][0].cpu().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 推送 transition
        transitions_q.put((obs, action, reward, next_obs, done))

        obs = next_obs if not done else env.reset()[0]

    print("[Actor] Exit")


# ===============================================================
# 2. Learner process
# ===============================================================
def run_learner(transitions_q, params_q, stop_event, policy, obs_features, act_features, device):

    print("[Learner] Started")

    buffer = ReplayBuffer(device=device, state_keys=list(obs_features.keys()))
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    step = 0

    while not stop_event.is_set():

        # 等待 actor 的 transition
        try:
            s, a, r, ns, done = transitions_q.get(timeout=1)
        except:
            continue

        buffer.add_transition(
            observation=s,
            action=a,
            reward=r,
            next_observation=ns,
            done=done,
        )

        if len(buffer) < 1024:
            continue

        batch = buffer.sample(batch_size=64)

        # SACPolicy 训练接口：forward + compute_loss（lerobot 自带）
        out = policy(**batch)
        loss = out.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        if step % 20 == 0:
            params_q.put(policy.state_dict())

        if step % 100 == 0:
            print(f"[Learner] step={step}, loss={loss.item():.4f}")

    print("[Learner] Exit")


# ===============================================================
# 3. Main
# ===============================================================
if __name__ == "__main__":

    mp.set_start_method("spawn")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_name = "gym_hil/PandaPickCubeKeyboard-v0"

    # 用一次环境获取 observation / action 空间特征
    env_tmp = gym.make(env_name, image_obs=False)
    print(env_tmp)
    print(env_tmp.__dict__)

    def get_env_layer(env, n):
        """Return the nth wrapper layer (1-based), i.e., env.env.env..."""
        current = env
        for _ in range(n-1):
            if hasattr(current, "env"):
                current = current.env
            else:
                return None
        return current


    env_real = get_env_layer(env_tmp, 9)
    print(env_real)
    print(env_real.__dict__)

    obs_features = hw_to_dataset_features(
        env_real.observation_space,
        prefix="observation"
    )

    action_features = hw_to_dataset_features(
        env_real.action_space,
        prefix="action"
    )

    obs_features = hw_to_dataset_features(obs_features, "observation")

    action_features = hw_to_dataset_features(action_features, "action")
    env_tmp.close()

    # 初始化 policy
    policy_cfg = SACConfig(
        device=device,
        input_features=obs_features,
        output_features=action_features,
    )

    policy_actor = SACPolicy(policy_cfg).to(device)
    policy_learner = SACPolicy(policy_cfg).to(device)

    # 创建多进程通信
    transitions_q = mp.Queue(maxsize=50)
    params_q = mp.Queue(maxsize=2)
    stop_event = mp.Event()

    def _handler(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handler)

    # actor 进程
    actor = mp.Process(
        target=run_actor,
        args=(transitions_q, params_q, stop_event, policy_actor, env_name, device),
    )

    learner = mp.Process(
        target=run_learner,
        args=(transitions_q, params_q, stop_event,
              policy_learner, obs_features, action_features, device),
    )

    actor.start()
    learner.start()

    print("MP SAC started! Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        stop_event.set()

    actor.join()
    learner.join()
