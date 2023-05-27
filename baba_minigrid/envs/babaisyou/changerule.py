from baba_minigrid.babaisyou import BabaIsYouEnv, BabaIsYouGrid
from baba_minigrid.flexible_world_object import FBall, FWall, Baba, RuleObject, RuleIs, RuleProperty
from baba_minigrid.utils import grid_random_position
from baba_minigrid.envs.babaisyou.goto import BaseGridEnv, random_rule_pos
from baba_minigrid.minigrid import Grid, MissionSpace, MiniGridEnv


class ChangeRuleEnv(MiniGridEnv):
    def __init__(self, size=8, agent_start_pos=(1, 1), agent_start_dir=0, **kwargs):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(
            mission_func=lambda: ""
        )

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = BabaIsYouGrid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        # self.put_obj(Goal(), width - 2, height - 2)
        self.put_obj(FBall(), width - 2, height - 2)

        self.grid.horz_wall(1, height-4, length=width-2, obj_type=FWall)
        # self.put_obj(RuleBlock("can_overlap", "fwall"), width-2, 1)

        self.put_obj(RuleObject('fwall'), 2, 2)
        # self.put_obj(RuleWall('fwall'), 2, 2)
        self.put_obj(RuleIs(), 3, 2)
        # self.put_obj(RuleProperty('can_overlap'), 5, 2)
        self.put_obj(RuleProperty('is_stop'), 4, 2)

        self.put_obj(RuleProperty('is_goal'), 5, height-3)
        # self.put_obj(RuleObject('fwall'), 3, height-3)

        self.put_obj(RuleObject('fball'), 2, height-3)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()


class TestRuleEnv(BabaIsYouEnv):
    def __init__(self, **kwargs):
        self.blocks = [
            RuleObject('fwall'),
            RuleObject('fball'),
            RuleIs(),
            RuleIs(),
            RuleIs(),
            RuleProperty('is_stop'),
            RuleProperty('is_goal'),
            RuleProperty('is_push'),
            RuleProperty('is_pull'),
            RuleProperty('is_defeat'),
            RuleProperty('is_move'),
            FBall(),
            FBall(),
            FBall(),
            FWall()
        ]
        self.size = 14
        super().__init__(grid_size=self.size, max_steps=int(1e5), **kwargs)

    def _gen_grid(self, width, height):
        self.grid = BabaIsYouGrid(width, height)
        # self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.put_obj(RuleObject('baba'), 2, 2)
        self.put_obj(RuleIs(), 3, 2)
        self.put_obj(RuleProperty('is_agent'), 4, 2)

        positions = grid_random_position(self.size, n_samples=len(self.blocks), margin=3)

        for pos, block in zip(positions, self.blocks):
            self.put_obj(block, *pos)

        self.place_obj(Baba())
        self.place_agent()


class MoveBlockEnv(BabaIsYouEnv):
    def __init__(self, **kwargs):
        self.goal_pos = (3, 4)
        self.goal_obj = RuleProperty('is_goal')
        self.size = kwargs.get('size', 7)
        super().__init__(grid_size=self.size, max_steps=4*self.size*self.size, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.put_obj(RuleObject('baba'), 1, 1)
        self.put_obj(RuleIs(), 2, 1)
        self.put_obj(RuleProperty('is_agent'), 3, 1)

        self.place_obj(self.goal_obj, top=(2, 2), size=[self.size-4, self.size-4])

        self.place_obj(Baba())
        self.place_agent()

    def reward(self):
        if self.grid.get(*self.goal_pos) == self.goal_obj:
            return self.get_reward(), True
        else:
            return 0, False


class MakeRuleEnv(BabaIsYouEnv):
    def __init__(self, **kwargs):
        self.rule_obj= RuleObject('fball')
        self.rule_is = RuleIs()
        self.rule_prop = RuleProperty('is_goal')
        self.size = kwargs.get('size', 7)
        super().__init__(grid_size=self.size, max_steps=4*self.size*self.size, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.put_obj(RuleObject('baba'), 1, 1)
        self.put_obj(RuleIs(), 2, 1)
        self.put_obj(RuleProperty('is_agent'), 3, 1)

        self.place_obj(self.rule_obj, top=(2, 2), size=[self.size-4, self.size-4])
        self.place_obj(self.rule_is, top=(2, 2), size=[self.size-4, self.size-4])
        self.place_obj(self.rule_prop, top=(2, 2), size=[self.size-4, self.size-4])

        self.place_obj(Baba())
        self.place_agent()

    def reward(self):
        ruleset = self.get_ruleset()
        if ruleset['is_goal'].get('fball', False):
            return self.get_reward(), True
        else:
            return 0, False
