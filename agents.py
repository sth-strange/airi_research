import numpy as np
from typing import Tuple
import random


class BaseAgent:
    def __init__(self, agent_name: str = "", agent_type: str = "", location: Tuple = (), key: bool = False):
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.location = location
        self.start_zone = []
        self.status = ""

    def get_q(self, state, action):
        pass

    def update_q(self, state, action, reward, next_state):
        pass

    def select_action(self, state):
        pass

    def decay_epsilon(self):
        pass


class Patron(BaseAgent):
    def __init__(
            self,
            action_space,
            alpha=0.1,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            min_epsilon=0.01,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.q_table = {}
        self.action_space = action_space
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q(self, state, action, reward, next_state):
        all_actions = range(self.action_space.n)
        best_next_action = max(all_actions, key=lambda a: self.get_q(next_state, a))
        td_target = reward + self.gamma * self.get_q(next_state, best_next_action)
        td_error = td_target - self.get_q(state, action)
        new_q = self.get_q(state, action) + self.alpha * td_error
        self.q_table[(state, action)] = new_q

    def select_action(self, agent_location):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()  # Exploration
        else:
            all_actions = range(self.action_space.n)
            return max(all_actions, key=lambda a: self.get_q(agent_location, a))  # Exploitation

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


class Altruist(BaseAgent):
    def __init__(
            self,
            action_space,
            alpha=0.5,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            min_epsilon=0.01,
            decay_coefficient = 0.95,
            time_horizon = 5,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.q_table = {}
        self.action_space = action_space
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.negative_reward = 0.1
        self.alpha_changing = alpha

        self.time = 0
        self.time_horizon = time_horizon
        self._action_to_direction = {
            0: np.array([0, -1]),
            1: np.array([1, 0]),
            2: np.array([0, 1]),
            3: np.array([-1, 0]),
            4: np.array([0, 0]),
        }
        self._direction_to_action = {
            (0, -1) : 0,
            (1, 0) : 1,
            (0, 1) : 2,
            (-1, 0) : 3,
            (0, 0): 4
        }
        self.decay_coefficient = decay_coefficient
        self.states_of_env = {}
        self.score_time = 0
        self.decay_epsilon_counter = 0
        self.step_number = 0
        

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q(self, state, action, reward, next_state):
        all_actions = range(self.action_space.n)
        score = 0
        exp = 1
        self.score_time = self.time - self.time_horizon
        if self.score_time < 0:
            return
        reachable_tiles = [self.states_of_env[self.score_time]["patron_position"]]
        while(self.score_time < self.time):
            next_tiles = set()
            for tile in reachable_tiles:
                for direction in self._action_to_direction.values():
                    if self._allowed_step_patron(tile, direction):
                        next_tiles.add(tuple(direction + tile))
            score = score + exp * len(next_tiles)
            self.score_time += 1
            exp = exp * self.decay_coefficient
            reachable_tiles = list(next_tiles)

        state = self.states_of_env[self.time - self.time_horizon]["altruist_position"]
        action_id = self.states_of_env[self.time - self.time_horizon]["altruist_action"]
        if self._allowed_step_altruist(state, self._action_to_direction[action_id]):
            new_q = self.get_q(state, action_id) + self.alpha_changing * score
        else:
            new_q = self.get_q(state, action_id) - self.alpha * self.negative_reward
        self.q_table[(state, action_id)] = new_q



    def select_action(self, agent_location, altruist_steps):
        self.time += 1
        if altruist_steps:
            self.step_number += 1
            if self.step_number >= len(altruist_steps):
                self.step_number = len(altruist_steps)
            step_0 = - int(altruist_steps[self.step_number-1][0][0]) + int(altruist_steps[self.step_number-1][-1][0])
            step_1 = - int(altruist_steps[self.step_number-1][0][-1]) + int(altruist_steps[self.step_number-1][-1][1])
            print("altruist: ",  step_0, step_1)
            if (step_0, step_1) not in self._direction_to_action.keys():
                return 4
            return self._direction_to_action[(step_0, step_1)]
    
        match self.status:
            case "random":
                return self.action_space.sample()
            case "training":
                if np.random.rand() < self.epsilon:
                    return self.action_space.sample()  # Exploration
                else:
                    all_actions = range(self.action_space.n)
                    return max(all_actions, key=lambda a: self.get_q(agent_location, a))  # Exploitation
            case _:
                return self.action_space.sample()


    def _allowed_step_altruist(self, agent_location, direction):
        new_position = self.decision_grid_edges(agent_location, direction)
        if self.decision_other_agents_by_altruist(new_position):
            return True
        return False

    def _allowed_step_patron(self, agent_location, direction):
        new_position = self.decision_grid_edges(agent_location, direction)
        if not np.array_equal(new_position, agent_location + direction):
            return False
        if (self.decision_walls_positions(new_position)
                and self.decision_doors_positions(new_position)
                and self.decision_other_agents_by_patron(new_position)):
            return True
        return False

    def decision_grid_edges(self, agent_location, direction):
        new_position = np.clip(
            agent_location + direction, [0, 0], [self.states_of_env["length_of_grid"] - 1, self.states_of_env["height_of_grid"] - 1]
        )
        return new_position

    def decision_walls_positions(self, new_position):
        if tuple(new_position) in self.states_of_env["walls_positions"]:
            return False
        return True

    def decision_doors_positions(self, new_position):
        button_coords = self.states_of_env["doors_positions"].get(tuple(new_position), False)
        if not button_coords:
            return True
        if button_coords == self.states_of_env[self.score_time + 1]["altruist_position"]:
            return True
        return False

    def decision_other_agents_by_patron(self, new_position):
        if tuple(new_position) == self.states_of_env[self.score_time + 1]["altruist_position"]:
            return False
        return True

    def decision_other_agents_by_altruist(self, new_position):
        if tuple(new_position) == self.states_of_env[self.time - self.time_horizon]["patron_position"]:
            return False
        return True

    def decay_epsilon(self):
        self.decay_epsilon_counter += 1
        if self.decay_epsilon_counter > 300:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)