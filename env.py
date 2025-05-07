import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from visualization import GridRenderer
import functools


def _get_info():
    # Возвращаем дополнительную информацию, если необходимо
    return {}


class WorldEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
        "name": "v0",
    }
    scenary_type: str
    progon_number: int

    def __init__(self, scenary_type, size_x, size_y, target_location, walls_positions, doors_positions, progon_number=None, render_mode=None):
        self.render_mode = render_mode
        self.scenary_type = scenary_type
        self.progon_number = progon_number
        self.agents = {}
        self.size_x = size_x
        self.size_y = size_y
        self.progon_number = None
        self.target_location = target_location
        self.walls_positions = walls_positions
        self.doors_positions = doors_positions
        self._action_to_direction = {
            0: np.array([0, -1]),
            1: np.array([1, 0]),
            2: np.array([0, 1]),
            3: np.array([-1, 0]),
            4: np.array([0, 0])
        }

    def _get_obs(self):
        state = {"target": self.target_location}
        for agent_id, agent_instance in self.agents.items():
            state[agent_id] = agent_instance.location
        return state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        for agent_id, agent_instance in self.agents.items():
            agent_instance.location = random.choice(agent_instance.start_zone)
        observation = self._get_obs()
        info = _get_info()
        return observation, info

    def step(self, action):
        # Официально больше не поддерживаем множественное число патронов/альтруистов
        terminated = False
        direction = {}
        for agent_id in self.agents:
            direction[agent_id] = self._action_to_direction[action[agent_id]]
        if "altruist_0" in self.agents:
            agent_id = "altruist_0"
            agent_instance = self.agents[agent_id]
            agent_instance.location = self.altruist_decision_process(agent_instance, direction[agent_id])
        agent_id = "patron_0"
        agent_instance = self.agents[agent_id]
        agent_instance.location = self.patron_decision_process(agent_instance, direction[agent_id])
        if np.array_equal(agent_instance.location, self.target_location):
            terminated = True
        reward = 1 if terminated else -0.2
        observation = self._get_obs()
        info = _get_info()
        return observation, reward, terminated, False, info

    def altruist_decision_process(self, agent_instance, direction):
        new_position = self.decision_grid_edges(agent_instance, direction)
        if (self.decision_walls_positions(new_position)
                and self.decision_doors_positions(new_position)
                and self.decision_other_agents_by_altruist(new_position)):
            if self.render_mode == "human":
                self.check_for_door_buttons(new_position)
            return tuple(new_position)
        return agent_instance.location
    
    def patron_decision_process(self, agent_instance, direction):
        new_position = self.decision_grid_edges(agent_instance, direction)
        if (self.decision_walls_positions(new_position)
                and self.decision_doors_positions(new_position)
                and self.decision_other_agents_by_patron(new_position)):
            return tuple(new_position)
        return agent_instance.location

    def decision_grid_edges(self, agent_instance, direction):
        new_position = np.clip(
            agent_instance.location + direction, [0, 0], [self.size_x - 1, self.size_y - 1]
        )
        return new_position

    def decision_walls_positions(self, new_position):
        if tuple(new_position) in self.walls_positions:
            return False
        return True

    def decision_doors_positions(self, new_position):
        button_coords = self.doors_positions.get(tuple(new_position), False)
        if not button_coords:
            return True
        if "altruist_0" in self.agents:
            if button_coords == self.agents["altruist_0"].location:
                return True
        return False

    def decision_other_agents_by_patron(self, new_position):
        if "altruist_0" in self.agents:
            if tuple(new_position) == self.agents["altruist_0"].location:
                return False
        return True
    
    def decision_other_agents_by_altruist(self, new_position):
        if "patron_0" in self.agents:
            if tuple(new_position) == self.agents["patron_0"].location:
                return False
        return True
    
    def check_for_door_buttons(self, new_position):
        self.renderer.pushed_buttons.clear()
        if tuple(new_position) in self.doors_positions.values():
            self.renderer.pushed_buttons.add(tuple(new_position))
        return

    def render(self, step_number, episod_number):
        if self.render_mode == "human":
            # Отрисовываем только при вызове этого метода
            self.renderer.render(self.agents.values(), self.target_location, self.walls_positions, self.doors_positions, step_number, episod_number)

    def close(self):
        if self.render_mode == "human":
            self.renderer.close()
    
    @property
    def render_mode(self):
        return self._render_mode

    @render_mode.setter
    def render_mode(self, render_mode):
        if render_mode is not None and render_mode not in self.metadata.get("render_modes", []):
            raise ValueError(f"Invalid render_mode '{render_mode}'. Supported modes: {self.metadata['render_modes']}")
        self._render_mode = render_mode
        if render_mode == "human":
            self.renderer = GridRenderer(
                grid_width=self.size_x,
                grid_height=self.size_y,
                scenary_type=self.scenary_type,
                progon_number=self.progon_number
            )

    @functools.lru_cache(maxsize=None)
    def observation_space(self):
        return spaces.MultiDiscrete([self.size_x, self.size_y])

    @functools.lru_cache(maxsize=None)
    def action_space(self):
        return spaces.Discrete(4)
