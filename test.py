import pygame
import torch
import numpy as np
from pong_env import Pong
from deep_q_network import DeepQNetwork


def test(model_path="models/pong_model_final.pth", render=True):# 更改model_path，选择不同的模型权重
    pygame.init()

    # 创建窗口用于显示
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Pong Test")

    env = Pong()
    model = DeepQNetwork()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    state = env.reset()
    done = False
    total_reward = 0

    clock = pygame.time.Clock()

    while not done:
        # 事件处理，允许关闭窗口
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = model(state_tensor).argmax().item()

        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state

        if render:
            env.render(screen)
            clock.tick(60)  # 控制帧率

    print(f"Test episode total reward: {total_reward}")
    pygame.quit()


if __name__ == "__main__":
    test()
