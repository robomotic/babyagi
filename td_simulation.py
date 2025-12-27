import numpy as np
import time
import json

class CombinationLockMDP:
    def __init__(self, S=10, k=4):
        self.S = S
        self.k = k
        self.reset()
    def reset(self, start_state=0):
        self.state = start_state
        return self.state
    def step(self, action):
        if action == 1:
            self.state += 1
            if self.state == self.k: return self.state, 1.0, True
            return self.state, 0.0, False
        else:
            self.state = 0
            return self.state, 0.0, False

class QLearningAgent:
    def __init__(self, S, k, alpha=0.5, gamma=0.9, lambd=0.0, epsilon=0.1):
        self.S, self.k, self.alpha, self.gamma, self.lambd, self.epsilon = S, k, alpha, gamma, lambd, epsilon
        self.q = np.zeros((k + 1, S))
        self.e = np.zeros((k + 1, S))
    def choose_action(self, s):
        if np.random.random() < self.epsilon: return np.random.randint(self.S)
        return np.argmax(self.q[s])
    def update(self, s, a, r, s_next, terminal):
        a_next_best = np.argmax(self.q[s_next])
        target = r + (0 if terminal else self.gamma * self.q[s_next, a_next_best])
        delta = target - self.q[s, a]
        if self.lambd == 0:
            self.q[s, a] += self.alpha * delta
        else:
            self.e[s, a] += 1
            for st in range(self.k + 1):
                for act in range(self.S):
                    self.q[st, act] += self.alpha * delta * self.e[st, act]
                    if terminal: self.e[st, act] = 0
                    else: self.e[st, act] *= self.gamma * self.lambd

def run_experiment(agent_type, S, k, total_episodes=12000):
    env = CombinationLockMDP(S, k)
    agent = QLearningAgent(S, k, lambd=(0.9 if 'lambda' in agent_type else 0.0))
    metrics = {"eps_to_first_signal": None, "eps_to_mastery": None}
    
    if 'scaffolded' in agent_type:
        phase_eps = total_episodes // k
        for phase in range(k):
            start_state = k - 1 - phase
            for e in range(phase_eps):
                s = env.reset(start_state)
                done = False
                while not done:
                    a = agent.choose_action(s)
                    s_next, r, done = env.step(a)
                    agent.update(s, a, r, s_next, done)
                    s = s_next
                    if done or s < start_state: break
                curr_total = phase * phase_eps + e
                if metrics["eps_to_first_signal"] is None and np.max(agent.q[k-1]) > 0.01: metrics["eps_to_first_signal"] = curr_total
                if metrics["eps_to_mastery"] is None and np.max(agent.q[0]) > 0.1: metrics["eps_to_mastery"] = curr_total
    else:
        for e in range(total_episodes):
            s = env.reset(0)
            done = False
            while not done:
                a = agent.choose_action(s)
                s_next, r, done = env.step(a)
                agent.update(s, a, r, s_next, done)
                s = s_next
                if s == 0 and a != 1: break
            if metrics["eps_to_first_signal"] is None and np.max(agent.q[k-1]) > 0.01: metrics["eps_to_first_signal"] = e
            if metrics["eps_to_mastery"] is None and np.max(agent.q[0]) > 0.1: metrics["eps_to_mastery"] = e
    return metrics

def main():
    S, k = 10, 4
    print(f"Final Proof Simulation: Alphabet S={S}, Sequence k={k}")
    for name in ['classic_td0', 'classic_td_lambda', 'scaffolded_td0']:
        m = run_experiment(name, S, k)
        print(f"{name:20} -> Signal: {str(m['eps_to_first_signal']):8} | Mastery: {str(m['eps_to_mastery']):8}")

if __name__ == "__main__":
    main()
