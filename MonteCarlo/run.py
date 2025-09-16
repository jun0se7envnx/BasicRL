import numpy as np
import gymnasium as gym
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2
import time

if __name__ == "__main__":

    # Tham số
    EPISODES = 5000
    GAMMA = 0.99
    epsilon = 0.9
    MAX_STEPS = 20

    # Tạo environment
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="rgb_array")

    # Q-table và returns
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns = defaultdict(list)

    def epsilon_greedy_policy(state, epsilon=0.9):
        if random.random() < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(Q[state])

    episode_rewards = []

    for episode in range(EPISODES):
        state, _ = env.reset()
        trajectory = []
        total_reward = 0

        if episode % 10 == 0:
            epsilon -= 0.01

        # Sinh một episode
        for _ in range(MAX_STEPS):
            action = epsilon_greedy_policy(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            trajectory.append((state, action, reward))
            total_reward += reward
            if terminated or truncated:
                break
            state = next_state

        episode_rewards.append(total_reward)

        # Monte Carlo - First Visit
        G = 0
        visited = set()
        for t in reversed(range(len(trajectory))):
            state_t, action_t, reward_t = trajectory[t]
            G = GAMMA * G + reward_t
            if (state_t, action_t) not in visited:
                visited.add((state_t, action_t))
                returns[(state_t, action_t)].append(G)
                Q[state_t][action_t] = np.mean(returns[(state_t, action_t)])

    def moving_avg(data, window=100):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    state, _ = env.reset()
    for step in range(MAX_STEPS):
        # Lấy hành động tốt nhất
        action = np.argmax(Q[state])
        
        # Thực hiện hành động
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # Lấy ảnh render (mảng numpy HxWx3)
        frame = env.render()
        

        # cv2 dùng BGR nên chuyển từ RGB sang BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(frame_bgr.shape)
        # Hiển thị ảnh
        cv2.imshow('FrozenLake Policy Play', frame_bgr)
        
        # Delay ms, thoát nếu nhấn phím 'q'
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
        
        if terminated or truncated:
            print(f"Episode finished after {step+1} steps with reward {reward}")
            time.sleep(5)  # đợi 1s để xem màn hình cuối
            break
        
        state = next_state
    
    cv2.destroyAllWindows()

    env.close()
