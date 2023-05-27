import numpy as np
from matplotlib import pyplot as plt

from baba_minigrid.babaisyou import BabaIsYouGrid
from baba_minigrid.envs.babaisyou.core.flexible_world_object import FBall, Baba
from baba_minigrid.envs.babaisyou.goto import BaseGridEnv


class TestEnv(BaseGridEnv):
    def __init__(self, ball_pos, baba_pos, size=8, **kwargs):
        self.ball_pos = ball_pos
        self.baba_pos = baba_pos
        self.size = size
        super().__init__(size=size, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = BabaIsYouGrid(width, height, )
        self.grid.wall_rect(0, 0, width, height)

        self.put_obj(FBall(), *self.ball_pos) if self.ball_pos is not None else None
        self.put_obj(Baba(), *self.baba_pos) if self.baba_pos is not None else None
        self.place_agent()


def test_grid_encoding1():
    def test_stacked_obj(encoding_level):
        env1 = TestEnv(ball_pos=(5, 4), baba_pos=(5, 4), encoding_level=encoding_level)
        env2 = TestEnv(ball_pos=None, baba_pos=(5, 4), encoding_level=encoding_level)
        # env2 = TestEnv(ball_pos=(5, 4), baba_pos=None, encoding_level=encoding_level)
        obs1, _, _, _ = env1.step(env1.actions.idle)
        obs2, _, _, _ = env2.step(env2.actions.idle)
        return obs1, obs2

    # [13  6  0 11  1  0] baba, ball
    # [11  1  0  1  0  0] ball, empty
    # [13  6  0  1  0  0] baba, empty

    obs1, obs2 = test_stacked_obj(encoding_level=1)
    # assert np.allclose(obs1, obs2)
    obs1, obs2 = test_stacked_obj(encoding_level=2)
    # assert not np.allclose(obs1, obs2)

    print(obs1.shape)
    pos = (5, 4)
    print(obs1[pos].shape)
    print(obs1[pos])
    print(obs2[pos])

    # print(obs[5, 4])


    # def plot(obs):
    #     obs = obs / np.max(obs)
    #     plt.imshow(obs)
    #     plt.show()
    #
    # plot(obs1)
    # plot(obs2)