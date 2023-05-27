import numpy as np

from baba_minigrid import Wall
from baba_minigrid.babaisyou import BabaIsYouEnv, BabaIsYouGrid
from baba_minigrid.flexible_world_object import RuleObject, RuleIs, RuleProperty, Baba, make_obj, \
    FDoor, FWall, FBall, FKey
from baba_minigrid.babaisyou import put_rule


class MoveObjEnv(BabaIsYouEnv):
    def __init__(self, obj="fball", goal_pos='random', size=7, **kwargs):
        self.goal_pos = goal_pos

        obj = [obj] if not isinstance(obj, list) else obj
        self.goal_obj_name = obj
        self.goal_obj = [make_obj(o) for o in obj]

        self.size = size
        super().__init__(grid_size=self.size, max_steps=4*self.size*self.size, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.put_obj(RuleObject('baba'), 1, 1)
        self.put_obj(RuleIs(), 2, 1)
        self.put_obj(RuleProperty('is_agent'), 3, 1)

        idx = np.random.choice(range(len(self.goal_obj)))
        sampled_obj_name, self.sampled_obj = self.goal_obj_name[idx], self.goal_obj[idx]

        self.put_obj(RuleObject(sampled_obj_name), 1, 2)
        self.put_obj(RuleIs(), 2, 2)
        self.put_obj(RuleProperty('is_push'), 3, 2)

        if self.goal_pos == 'random':
            self.current_goal_pos = self.place_obj(FDoor())
        else:
            self.put_obj(FDoor(), *self.goal_pos)
            self.current_goal_pos = self.goal_pos
        for o in self.goal_obj:
            self.place_obj(o, top=(2, 2), size=[self.size-4, self.size-4])

        self.place_obj(Baba())
        self.place_agent()

    def reward(self):
        if self.grid.get(*self.current_goal_pos) == self.sampled_obj:
            return self.get_reward(), True
        else:
            return 0, False


class OpenShutObjEnv(BabaIsYouEnv):
    object_names = ["fball", "fwall", "fkey", "fdoor"]

    def __init__(self, open_objects: list[str] = None, shut_objects: list[str] = None, size=8, **kwargs):
        open_objects = self.object_names if open_objects is None else open_objects
        shut_objects = self.object_names if shut_objects is None else shut_objects
        self.open_objects = {name: make_obj(name) for name in open_objects}
        self.shut_objects = {name: self.open_objects.get(name, make_obj(name)) for name in shut_objects}
        self.all_objects = {**self.open_objects, **self.shut_objects}

        default_ruleset = {
            "is_agent": {"baba": True},
            # "is_push": {obj: True for obj in self.object_names}
        }

        self.size = size
        super().__init__(grid_size=self.size, max_steps=4*self.size*self.size, default_ruleset=default_ruleset, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # sample open and shut objects
        open_obj_name = np.random.choice(list(self.open_objects.keys()))
        # ensure that shut obj is different from open obj
        shut_objects = dict(self.shut_objects)
        if open_obj_name in shut_objects:
            del shut_objects[open_obj_name]
        shut_obj_name = np.random.choice(list(shut_objects.keys()))

        self.open_obj = self.open_objects[open_obj_name]
        self.shut_obj = self.shut_objects[shut_obj_name]

        # open and push rules
        self.put_obj(RuleObject(open_obj_name), 1, 1)
        self.put_obj(RuleIs(), 2, 1)
        self.put_obj(RuleProperty('is_push'), 3, 1)
        # self.put_obj(RuleProperty('is_open'), 3, 1)

        # self.put_obj(RuleObject(open_obj_name), 4, 1)
        # self.put_obj(RuleIs(), 5, 1)
        # self.put_obj(RuleProperty('is_push'), 6, 1)

        # shut rule
        self.put_obj(RuleObject(shut_obj_name), 1, 2)
        self.put_obj(RuleIs(), 2, 2)
        self.put_obj(RuleProperty('is_shut'), 3, 2)

        for name, obj in self.all_objects.items():
            pos = self.place_obj(obj, top=(2, 2), size=[self.size-4, self.size-4])
            if name == open_obj_name:
                self.open_obj_pos = pos
            elif name == shut_obj_name:
                self.shut_obj_pos = pos
        self.place_obj(Baba())
        self.place_agent()

    def reward(self):
        if self.grid.get(*self.shut_obj_pos) == self.open_obj:
            return self.get_reward(), True
        else:
            return 0, False


class OpenAndGoToWinEnv(BabaIsYouEnv):
    tasks = ["goto_win", "open_shut", "make_rule"]

    def __init__(self, separating_walls=True, task="goto_win", show_shut_obj=True, push_rule=True, height=8, width=11, **kwargs):
        self.metadata["render.modes"].append("text")

        self.separating_walls = separating_walls
        default_ruleset = {
            "is_agent": {"baba": True},
            "is_open": {"fkey": True},
            "is_stop": {"fdoor": True}
        }

        assert task in self.tasks, task
        self.task = task

        self.push_rule = push_rule  # don't activate the push rule if false
        self.show_shut_obj = show_shut_obj
        super().__init__(width=width, height=height, max_steps=4*width*height,
                         default_ruleset=default_ruleset, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        if self.separating_walls:
            self.grid.vert_wall(width//2, 1, length=height-2, obj_type=FWall)

        open_obj_name = "fkey"
        shut_obj_name = "fdoor"
        win_obj_name = "fball"

        if self.push_rule:
            put_rule(self, open_obj_name, 'is_push', [(1, 1), (2, 1), (3, 1)])
        else:
            put_rule(self, open_obj_name, 'is_push', [(1, 1), (2, 1), (3, 2)])

        put_rule(self, win_obj_name, 'is_goal', [(width-4, 1), (width-3, 1), (width-2, 1)])
        put_rule(self, shut_obj_name, 'is_shut', [(1, height-2), (2, height-2), (3, height-2)])
        put_rule(self, "fwall", 'is_stop', [(width-4, height-2), (width-3, height-2), (width-2, height-2)])

        self.place_obj(make_obj(open_obj_name), top=(3, 3), size=[width//2-4, height-6])

        shut_obj_pos = (width//2, height//2)
        self._shut_obj_pos = shut_obj_pos
        self.grid.set(width//2, height//2, None)  # first remove the wall
        if self.show_shut_obj:
            self.put_obj(make_obj(shut_obj_name), *shut_obj_pos)

        self.place_obj(make_obj(win_obj_name))

        # for name, obj in self.all_objects.items():
        #     pos = self.place_obj(obj, top=(2, 2), size=[self.size-4, self.size-4])
        #     if name == open_obj_name:
        #         self.open_obj_pos = pos
        #     elif name == shut_obj_name:
        #         self.shut_obj_pos = pos

        self.place_obj(Baba(), size=[width//2, height])
        self.place_agent()

    def reward(self):
        if self.task == "open_shut":
            assert self._shut_obj_pos is not None
            # check if the shut obj is destroyed
            if self.grid.get(*self._shut_obj_pos) is None:
                return self.get_reward(), True
            else:
                return 0, False

        elif self.task == "goto_win":
            return super().reward()

        elif self.task == "make_rule":
            ruleset = self.get_ruleset()
            if ruleset['is_push'].get('fkey', False):
                return self.get_reward(), True
            else:
                return 0, False

        else:
            raise ValueError(self.task)

    def get_obj_pos(self, obj_type):
        for j in range(self.grid.height):
            for i in range(self.grid.width):
                c = self.grid.get(i, j)
                if type(c) == obj_type:
                    return i, j
        return None

    def render(self, mode="human", **kwargs):
        if mode == "text":
            s = ""
            s += "Active rules:"
            s += "\n"
            ruleset = self.get_ruleset()
            for prop in ruleset.keys():
                for obj in ruleset[prop].keys():
                    if ruleset[prop][obj]:
                        _prop = prop.split("_")[1]
                        _obj = obj[1:] if obj[0] == 'f' else obj
                        _prop = "win" if _prop == "goal" else _prop  # TODO
                        s += f"- {_obj} is {_prop}"
                        s += "\n"

            # indicate if the rule "key is push" is not active (TODO: generally describe potential rules to form)
            if not ruleset['is_push'].get('fkey', False):
                s += "Unactive rules:" + "\n"
                s += "- key is push" + "\n"

            def describe_obj_pos(obj_type, obj_name):
                pos = self.get_obj_pos(obj_type)
                s = ""
                if pos is None:
                    s += ""
                elif pos[1] < self.width//2:
                    s += f"The {obj_name} is on the left side"
                else:
                    s += f"The {obj_name} is on the right side"
                return s

            s += describe_obj_pos(Baba, "agent") + "\n"
            s += describe_obj_pos(FBall, "ball") + "\n"
            s += describe_obj_pos(FKey, "key") + "\n"

            # shut obj
            if self.grid.get(*self._shut_obj_pos) is None:
                s += "There is no object between the left and right side"
            else:
                s += "There is a door between the left and right side"
            s += "\n"

            return s

        else:
            return super().render(mode=mode, **kwargs)


class FourRoomEnv(BabaIsYouEnv):
    def __init__(self, open_shut_task=False, show_shut_obj=True, height=15, width=15, randomize=False, objects=None,
                 rule_is_push=True, **kwargs):
        # default_ruleset = {
            # "is_agent": {"baba": True},
            # "is_open": {"fkey": True},
            # "is_stop": {"fdoor": True}  # TODO: bug with wall is stop
        # }
        self.objects = ["fkey", "fdoor", "fball", "fwall"] if objects is None else objects
        self.rule_is_push = rule_is_push  # can push rule blocks if true
        default_ruleset = {}
        self.open_shut_task = open_shut_task
        self.show_shut_obj = show_shut_obj
        self.randomize = randomize
        super().__init__(width=width, height=height, max_steps=2000,
                         default_ruleset=default_ruleset, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.grid.vert_wall(width//2, 1, length=height-2, obj_type=Wall)
        self.grid.horz_wall(1, height//2, length=width-2, obj_type=Wall)

        # TODO
        objects = self.objects
        rule_objects = ["baba", "fkey", "fdoor", "fkey"]

        def _permute(arr):
            indices = np.random.choice(np.arange(len(arr)), size=len(arr), replace=False)
            arr = np.array(arr)[indices]
            return arr

        if self.randomize:
            objects = _permute(objects)
            rule_objects = _permute(objects)

        put_rule(self, objects[0], 'is_push', [(2+i, 2) for i in range(3)], is_push=self.rule_is_push)
        put_rule(self, "baba", 'is_agent', [(2+i, 4) for i in range(3)], is_push=self.rule_is_push)

        put_rule(self, objects[1], 'is_shut', [(2+i, height-5) for i in range(3)], is_push=self.rule_is_push)
        put_rule(self, rule_objects[1], 'is_open', [(2+i, height-3) for i in range(3)], is_push=self.rule_is_push)

        put_rule(self, objects[2], 'is_goal', [(width-5+i, 2) for i in range(3)], is_push=self.rule_is_push)
        put_rule(self, rule_objects[2], 'is_stop', [(width-5+i, 4) for i in range(3)], is_push=self.rule_is_push)

        put_rule(self, objects[3], 'is_stop', [(width-5+i, height-3) for i in range(3)], is_push=self.rule_is_push)
        put_rule(self, rule_objects[3], "is_defeat", [(width-5+i, height-5) for i in range(3)], is_push=self.rule_is_push)

        def _put_obj(name, pos):
            self.grid.set(*pos, None)  # first remove the wall
            self.put_obj(make_obj(name), *pos)

        # self._shut_obj_pos = shut_obj_pos = (width//2, height//4)
        object_pos = [(width//2, height//4), (width//4, height//2), (3*width//4, height//2), (width//2, 3*height//4)]
        if self.randomize:
            object_pos = _permute(object_pos)
        # _put_obj(objects[0], shut_obj_pos)
        # _put_obj(objects[1], (width//4, height//2))
        # _put_obj(objects[2], (3*width//4, height//2))
        # _put_obj(objects[3], (width//2, 3*height//4))

        for i in range(len(objects)):
            _put_obj(objects[i], object_pos[i])

        if self.randomize:
            self.place_obj(Baba(), size=[width, height])
        else:
            self.put_obj(Baba(), 4, 3)
        self.place_agent()

    def reward(self):
        # directly lose if no object is the agent
        ruleset = self.get_ruleset()
        if ruleset['is_agent'] == {}:
            return -1, True

        if self.open_shut_task:
            assert self._shut_obj_pos is not None
            # check if the shut obj is destroyed
            if self.grid.get(*self._shut_obj_pos) is None:
                return self.get_reward(), True
            else:
                return 0, False

        else:
            return super().reward()