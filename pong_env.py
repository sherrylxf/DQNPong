import pygame
import numpy as np

# 初始化参数
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 300
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60
BALL_SIZE = 10
PADDLE_SPEED = 5
BALL_SPEED = 4

class Pong:
    def __init__(self):
        pygame.init()
        self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        self.reset()

    def reset(self):
        # 球的位置和速度
        self.ball_x = SCREEN_WIDTH // 2
        self.ball_y = SCREEN_HEIGHT // 2
        self.ball_vx = BALL_SPEED * np.random.choice([-1, 1])
        self.ball_vy = BALL_SPEED * np.random.choice([-1, 1])

        # 左侧玩家（智能体）挡板位置
        self.paddle_y = SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2

        # 右侧挡板固定位置（简单AI）
        self.opponent_y = SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2

        # 得分情况
        self.done = False

        # 返回状态向量
        return self._get_state()

    def step(self, action):
        """
        action: 0 - 向上移动挡板
                1 - 不动
                2 - 向下移动挡板
        """

        # 更新玩家挡板位置
        if action == 0:
            self.paddle_y -= PADDLE_SPEED
        elif action == 2:
            self.paddle_y += PADDLE_SPEED

        # 边界限制
        self.paddle_y = np.clip(self.paddle_y, 0, SCREEN_HEIGHT - PADDLE_HEIGHT)

        # 简单的对手挡板跟随球的Y坐标
        if self.opponent_y + PADDLE_HEIGHT / 2 < self.ball_y:
            self.opponent_y += PADDLE_SPEED
        elif self.opponent_y + PADDLE_HEIGHT / 2 > self.ball_y:
            self.opponent_y -= PADDLE_SPEED
        self.opponent_y = np.clip(self.opponent_y, 0, SCREEN_HEIGHT - PADDLE_HEIGHT)

        # 更新球的位置
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        reward = 0
        self.done = False

        # 球碰上下边界反弹
        if self.ball_y <= 0 or self.ball_y >= SCREEN_HEIGHT - BALL_SIZE:
            self.ball_vy = -self.ball_vy

        # 球碰到左侧挡板
        if (self.ball_x <= PADDLE_WIDTH and
                self.paddle_y < self.ball_y < self.paddle_y + PADDLE_HEIGHT):
            self.ball_x = PADDLE_WIDTH  # 立即把球推到边界外
            self.ball_vx = abs(self.ball_vx)  # 向右走
            reward = 1

        # 球碰到右侧挡板
        if (self.ball_x >= SCREEN_WIDTH - PADDLE_WIDTH - BALL_SIZE and
                self.opponent_y < self.ball_y < self.opponent_y + PADDLE_HEIGHT):
            self.ball_x = SCREEN_WIDTH - PADDLE_WIDTH - BALL_SIZE  # 推离右挡板
            self.ball_vx = -abs(self.ball_vx)

        # 球出左边界，游戏结束，惩罚
        if self.ball_x < 0:
            reward = -10
            self.done = True

        # 球出右边界，重新发球（奖励0）
        if self.ball_x > SCREEN_WIDTH:
            reward = 0
            self.ball_x = SCREEN_WIDTH // 2
            self.ball_y = SCREEN_HEIGHT // 2
            self.ball_vx = BALL_SPEED * np.random.choice([-1, 1])
            self.ball_vy = BALL_SPEED * np.random.choice([-1, 1])

        return self._get_state(), reward, self.done

    def _get_state(self):
        # 状态向量: [球x, 球y, 球vx, 球vy, 玩家挡板y]
        return np.array([
            self.ball_x / SCREEN_WIDTH,
            self.ball_y / SCREEN_HEIGHT,
            self.ball_vx / BALL_SPEED,
            self.ball_vy / BALL_SPEED,
            self.paddle_y / (SCREEN_HEIGHT - PADDLE_HEIGHT)
        ], dtype=np.float32)

    def render(self, screen):
        # 需要传入一个pygame显示窗口surface才能显示
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 255, 255),
                         (0, int(self.paddle_y), PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.rect(screen, (255, 255, 255),
                         (SCREEN_WIDTH - PADDLE_WIDTH, int(self.opponent_y), PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.rect(screen, (255, 255, 255),
                         (int(self.ball_x), int(self.ball_y), BALL_SIZE, BALL_SIZE))
        pygame.display.flip()
