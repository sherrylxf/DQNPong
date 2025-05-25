import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
import pygame

from pong_env import Pong
from deep_q_network import DeepQNetwork

# ---- 参数配置 ---
# 1-1000轮训练
resume = False               # 不使用断点续训
start_epoch = 1             # 从第1轮开始
end_epoch = 1000            # 到第1000轮结束

# 1001-2000轮训练
#resume = True                  # 是否继续训练
#start_epoch = 1001 if resume else 1
#end_epoch = 2000              # 总训练轮次`

load_model_path = f"models/pong_model_{start_epoch - 1}.pth" if resume else None
metrics_excel_path = "metrics.xlsx"

# ---- 保存趋势图 ----
def save_plot(values, ylabel, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(values)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} Over Episodes')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# ---- 训练函数 ----
def train():
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Pong DQN Training")

    env = Pong()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepQNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    batch_size = 64
    gamma = 0.99
    epsilon = 0.1 if resume else 1.0  # 从文件继续训练时，初始 epsilon 可以更低
    final_epsilon = 0.01
    decay = 0.995

    replay_memory = deque(maxlen=50000)
    writer = SummaryWriter("runs/pong_dqn_resume" if resume else "runs/pong_dqn")

    os.makedirs("models", exist_ok=True)

    # ---- 如果断点续训，加载模型 ----
    if resume and os.path.exists(load_model_path):
        model.load_state_dict(torch.load(load_model_path))
        print(f"Loaded model from {load_model_path}")
    else:
        print("Starting training from scratch.")

    # ---- 历史数据 ----
    rewards_history = []
    loss_history = []
    epsilon_history = []
    q_value_history = []

    # ---- 主循环 ----
    for epoch in range(start_epoch, end_epoch + 1):
        state = env.reset()
        state = torch.FloatTensor(state).to(device)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    writer.close()
                    return

            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                with torch.no_grad():
                    q_values = model(state.unsqueeze(0))
                    action = torch.argmax(q_values).item()

            next_state, reward, done = env.step(action)
            next_state = torch.FloatTensor(next_state).to(device)

            replay_memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            steps += 1

            env.render(screen)

            if len(replay_memory) < batch_size:
                continue

            batch = random.sample(replay_memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.stack(states)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.stack(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

            q_values = model(states).gather(1, actions.unsqueeze(1))
            with torch.no_grad():
                next_q_values = model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * gamma * next_q_values

            loss = criterion(q_values, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epsilon = max(final_epsilon, epsilon * decay)

        # ---- 收集训练指标 ----
        with torch.no_grad():
            mean_q = model(state.unsqueeze(0)).mean().item()

        rewards_history.append(total_reward)
        epsilon_history.append(epsilon)
        q_value_history.append(mean_q)

        if 'loss' in locals():
            loss_history.append(loss.item())
            print(
                f"Epoch {epoch} | Reward: {total_reward} | Loss: {loss.item():.4f} | Epsilon: {epsilon:.3f} | Mean Q: {mean_q:.2f}")
            writer.add_scalar("Loss", loss.item(), epoch)
        else:
            loss_history.append(0.0)
            print(f"Epoch {epoch} | Reward: {total_reward} | Loss: N/A | Epsilon: {epsilon:.3f} | Mean Q: {mean_q:.2f}")

        print(f"Epoch {epoch} | Reward: {total_reward} | Loss: {loss.item():.4f} | Epsilon: {epsilon:.3f} | Mean Q: {mean_q:.2f}")

        # ---- 写入 TensorBoard ----
        writer.add_scalar("Reward", total_reward, epoch)
        writer.add_scalar("Loss", loss.item(), epoch)
        writer.add_scalar("Epsilon", epsilon, epoch)
        writer.add_scalar("Mean_Q", mean_q, epoch)

        # ---- 保存模型 ----
        if epoch % 1 == 0:
            torch.save(model.state_dict(), f"models/pong_model_{epoch}.pth")

    # 保存最终模型
    torch.save(model.state_dict(), f"models/pong_model_{end_epoch}.pth")
    writer.close()

    # 导出 Excel 表格
    df = pd.DataFrame({
        "Epoch": list(range(start_epoch, end_epoch + 1)),
        "Reward": rewards_history,
        "Loss": loss_history,
        "Epsilon": epsilon_history,
        "Mean_Q": q_value_history,
    })
    df.to_excel(metrics_excel_path, index=False)
    print(f"Metrics saved to {metrics_excel_path}")

    # 自动绘图
    save_plot(rewards_history, "Reward", "reward_curve.png")
    save_plot(loss_history, "Loss", "loss_curve.png")
    save_plot(epsilon_history, "Epsilon", "epsilon_curve.png")
    save_plot(q_value_history, "Mean Q Value", "qvalue_curve.png")

    pygame.quit()

if __name__ == "__main__":
    train()
