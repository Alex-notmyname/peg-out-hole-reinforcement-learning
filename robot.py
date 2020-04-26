import copy

MAX_FORCE = 30
MAX_TORQUE = 10
MAX_DISTANCE = 10
incre = 0.1
incre_a = 0.05
com = 0.001

reset_incre = 5


class Robot:

    def position_act(self, action_p, s):
        
        """get position movement"""

        state = copy.deepcopy(s)

        if action_p == 0:
            state[0] += incre

        elif action_p == 1:
            state[0] -= incre

        elif action_p == 2:
            state[0] += 0

        elif action_p == 3:
            state[1] += incre

        elif action_p == 4:
            state[1] -= incre

        elif action_p == 5:
            state[1] += 0

        elif action_p == 6:
            state[2] += incre

        elif action_p == 7:
            state[2] -= incre

        self.state_p = state

        return self.state_p

    def pose_act(self, action_p, s):

        """get pose movement"""

        state = copy.deepcopy(s)

        if action_p == 0:
            state[0] += incre_a
            state[1] += incre_a
            state[2] += com

        elif action_p == 1:
            state[0] += incre_a
            state[1] += 0
            state[2] -= com

        elif action_p == 2:
            state[0] += incre_a
            state[1] -= incre_a
            state[2] += com

        elif action_p == 3:
            state[0] += 0
            state[1] += incre_a
            state[2] -= com

        elif action_p == 4:
            state[0] += 0
            state[1] += 0
            state[2] += com

        elif action_p == 5:
            state[0] += 0
            state[1] -= incre_a
            state[2] -= com

        elif action_p == 6:
            state[0] -= incre_a
            state[1] += incre_a
            state[2] += com

        elif action_p == 7:
            state[0] -= incre_a
            state[1] += 0
            state[2] -= com

        elif action_p == 8:
            state[0] -= incre_a
            state[1] -= incre_a
            state[2] += com

        self.state_p = state

        return self.state_p











