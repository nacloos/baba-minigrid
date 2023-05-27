from baba_minigrid.babaisyou import BabaIsYouGrid
from baba_minigrid.envs.babaisyou.core.flexible_world_object import FBall, Baba, FWall, FDoor, FKey
from baba_minigrid.envs.babaisyou.goto import BaseGridEnv
from baba_minigrid.envs.babaisyou.wrappers import OpenShutReward
from baba_minigrid.play_game import play_babaisyou


class TestWinLoseEnv(BaseGridEnv):
    def __init__(self, size=8, is_lose=False, **kwargs):
        self.size = size
        self.is_lose = is_lose
        super().__init__(size=size, **kwargs)

    def _gen_grid(self, width, height):
        self.rule_pos = [(2, 2), (3, 2), (4, 2)]
        self.agent_start_pos = (4, 4)
        ball_pos = (5, 4)

        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.put_rule(obj='baba', property='is_agent', positions=[(1, 1), (2, 1), (3, 1)])
        if not self.is_lose:
            self.put_rule(obj='fball', property='is_goal', positions=self.rule_pos)
        else:
            self.put_rule(obj='fball', property='is_defeat', positions=self.rule_pos)

        self.put_obj(FBall(), *ball_pos)
        self.put_obj(Baba(), *self.agent_start_pos)
        self.place_agent()

class TestPushEnv(BaseGridEnv):

    def __init__(self, size=8, collision=1, **kwargs):
        self.size = size
        self.collision = collision
        super().__init__(size=size, **kwargs)

    def _gen_grid(self, width, height):
        '''
        1: 2 agents directly across from each other push on an object
        2: 2 agents adjacent to each other push on an object
        3: 1 object pushing into adjacent pushable objects
        4: 2 agnets push objects onto the same space at the same time results in a stack
        5: 2 agents pushing against a line of objects in opposite directions
        6: 1 agent pushes a line of objects while 1+ agents push from another direction
        '''
        # General Set Up
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.put_rule(obj='baba', property='is_agent', positions=[(1, 1), (2, 1), (3, 1)])
        self.put_rule(obj='fball', property='is_move', positions=[(4, 1), (5, 1), (6, 1)])
        self.put_rule(obj='fwall', property='can_push', positions=[(1, 2), (2, 2), (3, 2)])
            # change direction? right now just moves to the right and stops when hits wall
        self.wall_loc = []
        self.add_goal = True

        # Test Specific Set Up
        if self.collision == 1:
            self.agent_start_pos = (4, 5)
            ball_pos = (2, 5)
            self.wall_loc.append((3, 5))

        elif self.collision == 2:
            self.agent_start_pos = (5, 5)
            ball_pos = (4, 4)
            self.wall_loc.append((5, 4))
            self.add_goal = False

        elif self.collision == 3:
            self.agent_start_pos = (1, 5)
            ball_pos = (6, 6)
            self.wall_loc = [(2, 5), (3, 5)]

        elif self.collision == 4:
            self.agent_start_pos = (5, 6)
            ball_pos = (3, 4)
            self.wall_loc = [(5, 5), (4, 4)]

        elif self.collision == 5:
            self.agent_start_pos = (5, 5)
            ball_pos = (2, 5)
            self.wall_loc = [(3, 5), (4, 5)]

        elif self.collision == 6:
            self.agent_start_pos = (4, 6)
            ball_pos = (2, 5)
            self.wall_loc = [(3, 5), (4, 5)]

        # Configure Grid with Specifications
        if self.add_goal:
            self.put_rule(obj='fball', property='is_goal', positions=[(4, 2), (5, 2), (6, 2)])

        for loc in self.wall_loc:
            self.put_obj(FWall(), *loc)
        self.put_obj(FBall(), *ball_pos)
        self.put_obj(Baba(), *self.agent_start_pos)
        self.place_agent()


class TestPullEnv(BaseGridEnv):

    def __init__(self, size=8, collision=1, **kwargs):
        self.size = size
        self.collision = collision
        super().__init__(size=size, **kwargs)

    def _gen_grid(self, width, height):
        '''
        1: 2 agents directly across from each other pull on an object
        2: 2 agents adjacent to each other pull on an object
        3: 1 object pulling adjacent pushable objects
        4: 2 agents pulling objects onto same space at same time results in a stack
        5: 2 agents pulling objects in opposite directions collide part 1
        6: 2 agents pulling objects in opposite directions collide part 2
        7: 2 agents pulling objects in opposite directions collide part 3
        8: 1 agent pulling line of obj in one direction while 1+ agents pull in another direction

        '''
        # General Set Up
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.put_rule(obj='baba', property='is_agent', positions=[(1, 1), (2, 1), (3, 1)])
        self.put_rule(obj='fball', property='is_move', positions=[(1, 2), (2, 2), (3, 2)])
        # change direction?
        self.put_rule(obj='fwall', property='is_pull', positions=[(4, 2), (5, 2), (6, 2)])
        self.add_goal = True
        self.wall_loc = []


        # Test Specific Set Up
        if self.collision == 1:
            self.agent_start_pos = (2, 5)
            ball_pos = (4, 5)
            self.wall_loc.append((3, 5))

        elif self. collision == 2:
            self.agent_start_pos = (3, 5)
            ball_pos = (4, 4)
            self.wall_loc.append((3, 4))

        elif self.collision == 3:
            self.agent_start_pos = (3, 4)
            ball_pos = (1, 6)
            self.wall_loc = [(4, 4), (5, 4)]

        elif self.collision == 4:
            self.agent_start_pos = (4, 4)
            ball_pos = (4, 4)
            self.wall_loc = [(4, 3), (3, 4)]
            self.add_goal = False

        elif self.collision == 5:
            self.agent_start_pos = (4, 4)
            ball_pos = (3, 4)
            self.wall_loc = [(2, 4), (5, 4)]
            self.add_goal = False

        elif self.collision == 6:
            self.agent_start_pos = (5, 4)
            ball_pos = (3, 4)
            self.wall_loc = [(2, 4), (6, 4)]
            self.add_goal = False

        elif self.collision == 7:
            self.agent_start_pos = (2, 4)
            ball_pos = (5, 4)
            self.wall_loc = [(3, 4), (4, 4)]

        elif self.collision == 8:
            self.agent_start_pos = (3, 5)
            ball_pos = (5, 4)
            self.wall_loc = [(3, 4), (4, 4)]

        # Configure Grid with Specifications
        if self.add_goal:
            self.put_rule(obj='fball', property='is_goal', positions=[(4, 1), (5, 1), (6, 1)])
        self.put_obj(FBall(), *ball_pos)
        self.put_obj(Baba(), *self.agent_start_pos)
        for loc in self.wall_loc:
            self.put_obj(FWall(), *loc)
        self.place_agent()


def test_win():
    env = TestWinLoseEnv()
    env.reset()
    obs, reward, done, info = env.step(env.actions.right)
    assert done
    assert reward > 0


def test_lose():
    env = TestWinLoseEnv(is_lose=True)
    env.reset()
    obs, reward, done, info = env.step(env.actions.right)
    assert done
    assert reward < 0


# def test_encoding():
#     env = TestWinLoseEnv()
#     obs = env.reset()
#     print(obs.shape)
#     import matplotlib.pyplot as plt
#     import numpy as np
#     print(np.min(obs), np.max(obs))
#     plt.imshow(obs*10)
#     plt.show()

def test_push_case1():
    # Expected: No movement
    env = TestPushEnv()
    env.reset()
    env.step(env.actions.left)
    assert (isinstance(env.grid.get(2, 5), FBall))
    assert (isinstance(env.grid.get(3, 5), FWall))
    assert(isinstance(env.grid.get(4, 5), Baba))

def test_push_case2():
    # Expected: Not sure how to handle this case (reason for pass)
    env = TestPushEnv(collision=2)
    env.reset()
    pass

def test_push_case3():
    # Expected: Baba pushes both objects
    env = TestPushEnv(collision=3)
    env.reset()
    env.step(env.actions.right)
    assert (isinstance(env.grid.get(3, 5), FWall))
    assert (isinstance(env.grid.get(4, 5), FWall))
    assert(isinstance(env.grid.get(2, 5), Baba))

def test_push_case4():
    # Expected: Both Baba and Ball move, pushing the walls into a stack
    env = TestPushEnv(collision=4)
    env.reset()
    env.step(env.actions.up)
    assert (isinstance(env.grid.get(5, 5), Baba))
    assert (isinstance(env.grid.get(5, 4), FWall))
    assert(isinstance(env.grid.get(4, 4), FBall))
    assert(env.grid.get(5, 3) is None)
    assert(env.grid.get(6, 4) is None)

def test_push_case5():
    # Expected: no movement
    env = TestPushEnv(collision=5)
    env.reset()
    env.step(env.actions.left)
    assert(isinstance(env.grid.get(5, 5), Baba))
    assert(isinstance(env.grid.get(3, 5), FWall))
    assert(isinstance(env.grid.get(4, 5), FWall))
    assert(isinstance(env.grid.get(2, 5), FBall))

def test_push_case6():
    # Expected: thinking the agent should be able to push the object next to it bue unsure
    env = TestPushEnv(collision=6)
    env.reset()
    env.step(env.actions.up)
    pass # unsure how we want to handle this case
    # assert(isinstance(env.grid.get(4, 5), Baba)) #wall stacked under
    # assert(isinstance(env.grid.get(4, 4), FWall))
    # assert(isinstance(env.grid.get(3, 5), FBall))
    # env.step(env.actions.up)
    # assert (isinstance(env.grid.get(4, 4), Baba))  # wall stacked under
    # assert (isinstance(env.grid.get(4, 3), FWall))
    # assert (isinstance(env.grid.get(4, 5), FBall))
    # assert (isinstance(env.grid.get(5, 5), FWall))

def test_pull_case1():
    # Expected: no movement
    env = TestPullEnv(collision=1)
    env.reset()
    env.step(env.actions.left)
    assert (isinstance(env.grid.get(2, 5), Baba))
    assert (isinstance(env.grid.get(3, 5), FWall))
    assert (isinstance(env.grid.get(4, 5), FBall))

def test_pull_case2():
    # Expected: not sure how we want to handle this case
    env = TestPullEnv(collision=2)
    env.reset()
    assert (isinstance(env.grid.get(3, 5), Baba))
    assert (isinstance(env.grid.get(3, 4), FWall))
    assert (isinstance(env.grid.get(4, 4), FBall))
    env.step(env.actions.left)
    pass

def test_pull_case3():
    # Expected: Baba pulls both objects
    env = TestPullEnv(collision=3)
    env.reset()
    env.step(env.actions.left)
    assert (isinstance(env.grid.get(2, 4), Baba))
    assert (isinstance(env.grid.get(3, 4), FWall))
    assert (isinstance(env.grid.get(4, 4), FWall))

def test_pull_case4():
    # Expected: Baba and ball move in same time step, resulting in a wall stack
    env = TestPullEnv(collision=4)
    env.reset()
    env.step(env.actions.down)
    assert (isinstance(env.grid.get(4, 5), Baba))
    assert (isinstance(env.grid.get(4, 4), FWall))
    assert (isinstance(env.grid.get(5, 4), FBall))
    env.ste(env.actions.down)

def test_pull_case5():
    # Expected: no movement
    env = TestPullEnv(collision=5)
    env.reset()
    env.step(env.actions.left)
    assert (isinstance(env.grid.get(4, 4), Baba))
    assert (isinstance(env.grid.get(2, 4), FWall))
    assert (isinstance(env.grid.get(5, 4), FWall))
    assert (isinstance(env.grid.get(3, 4), FBall))

def test_pull_case6():
    # Expected: Ball and Baba stack (and from there baba and ball are trapped and must move up or down)
    env = TestPullEnv(collision=6)
    env.reset()
    env.step(env.actions.left)
    assert (isinstance(env.grid.get(4, 4), Baba))
    assert (isinstance(env.grid.get(5, 4), FWall))
    assert (isinstance(env.grid.get(3, 4), FWall))
    env.step(env.actions.up)
    assert (isinstance(env.grid.get(4, 4), FBall))


def test_pull_case7():
    # Expected: set of pulled objects are split in half and pulled in opposite directions
    env = TestPullEnv(collision=7)
    env.reset()
    env.step(env.actions.left)
    assert (isinstance(env.grid.get(1, 4), Baba))
    assert (isinstance(env.grid.get(2, 4), FWall))
    assert (isinstance(env.grid.get(6, 4), FBall))
    assert (isinstance(env.grid.get(5, 4), FWall))

def test_pull_case8():
    # Expected: Objects break apart at natural spot
    env = TestPullEnv(collision=8)
    env.reset()
    env.step(env.actions.down)
    assert (isinstance(env.grid.get(3, 6), Baba))
    assert (isinstance(env.grid.get(3, 5), FWall))
    assert (isinstance(env.grid.get(6, 4), FBall))
    assert (isinstance(env.grid.get(5, 4), FWall))

# test_push_case1()
# test_push_case2() #unsure how to handle
# test_push_case3()
# test_push_case4()
# test_push_case5()
# test_push_case6() #unsure how to handle

# test_pull_case1()
# test_pull_case2() #unsure how to handle
# test_pull_case3()
# test_pull_case4()
# test_pull_case5()
# test_pull_case6()
# test_pull_case7()
# test_pull_case8()


class TestOpenShutEnv(BaseGridEnv):
    def __init__(self, size=8, **kwargs):
        self.size = size
        super().__init__(size=size, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.put_rule(obj='baba', property='is_agent', positions=[(1, 1), (2, 1), (3, 1)])
        self.put_rule(obj='fdoor', property='is_shut', positions=[(1, 2), (2, 2), (3, 2)])
        self.put_rule(obj='fdoor', property='is_stop', positions=[(1, 3), (2, 3), (3, 3)])
        self.put_rule(obj='fkey', property='is_open', positions=[(4, 1), (5, 1), (6, 1)])
        self.put_rule(obj='fkey', property='is_push', positions=[(4, 2), (5, 2), (6, 2)])

        self.put_obj(Baba(), 3, 4)
        self.put_obj(FKey(), 4, 4)
        self.put_obj(FDoor(), 5, 4)
        self.place_agent()


def test_open_shut_case1():
    # TODO: test stacked open obj
    env = TestOpenShutEnv()
    env.step(action=env.actions.right)
    assert isinstance(env.grid.get(4, 4), Baba)
    assert env.grid.get(5, 4) is None


def test_open_shut_reward():
    env = TestOpenShutEnv()
    env = OpenShutReward(env)

    env.reset()
    obs, reward, done, info = env.step(action=env.actions.right)
    assert reward > 0
