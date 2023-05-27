from gym import Wrapper

from baba_minigrid.babaisyou import BabaIsYouEnv


class OpenShutReward(Wrapper):
    def __init__(self, env: BabaIsYouEnv):
        super().__init__(env)
        self.env = env
        self._shut_obj_pos = None

    def reset(self, *, seed=None, return_info=False, options=None):
        obs = self.env.reset(seed=seed, return_info=return_info, options=options)
        # get the position of the shut obj
        # TODO: not working if the shut obj moves
        shut_obj_pos = None
        for j in range(0, self.height):
            for i in range(0, self.width):
                obj = self.env.grid.get(i, j)
                if obj is not None and hasattr(obj, 'is_shut') and obj.is_shut():
                    shut_obj_pos = (i, j)

        assert shut_obj_pos is not None, "No shut object in the env"
        self._shut_obj_pos = shut_obj_pos
        return obs

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, done = self.reward()
        return observation, reward, done, info

    def reward(self):
        assert self._shut_obj_pos is not None
        # check if the shut obj is destroyed
        if self.grid.get(*self._shut_obj_pos) is None:
            return self.env.get_reward(), True
        else:
            return 0, False
