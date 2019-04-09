import os
import torch

import numpy as np
from gym import spaces
import vrep
from a2c_ppo_acktr.residual.initial_policy_model import InitialPolicy, train_nn

from a2c_ppo_acktr.vrep_utils import check_for_errors, VrepEnv

np.set_printoptions(precision=2, linewidth=200)  # DEBUG

dir_path = os.getcwd()
scene_path = dir_path + '/reach_over_wall.ttt'


class ReachOverWallEnv(VrepEnv):

    def solve_ik(self):
        _, path, _, _ = self.call_lua_function('solve_ik')
        num_path_points = len(path) // len(self.joint_handles)
        path = np.reshape(path, (num_path_points, len(self.joint_handles)))
        distances = np.array([path[i + 1] - path[i]
                              for i in range(0, len(path) - 1)])
        velocities = distances * 20  # Distances should be covered in 0.05s
        return path, velocities

    def get_demo_path(self):
        path, velocities_WP = self.solve_ik()
        path_to_WP = path[:-1]

        self.call_lua_function('set_joint_angles', ints=self.init_config_tree, floats=path[-1])
        path_to_trg, velocities_trg = self.solve_ik()

        return self.normalise_angles(np.append(path_to_WP, path_to_trg[:-1], axis=0)), \
            np.append(velocities_WP, velocities_trg, axis=0)

    def train_initial_policy(self):
        x, y = self.get_demo_path()

        self.initial_policy = train_nn(InitialPolicy(len(x[0]), len(y[0])), x, y)

        null_action = np.array([0.] * len(y[0]))
        # Use DAgger for 5 episodes
        for i in range(5):
            self.reset()
            done = False
            while not done:
                _, _, done, _ = self.step(null_action)
                new_x, new_y = self.solve_ik()
                if len(new_x) != 0:
                    x = np.append(x, self.normalise_angles(new_x[:-1]), axis=0)
                    y = np.append(y, new_y, axis=0)

            p = self.np_random.permutation(len(x))
            self.initial_policy = train_nn(self.initial_policy, x[p], y[p])

    observation_space = spaces.Box(np.array([0] * 12), np.array([1] * 12),
                                   dtype=np.float32)
    action_space = spaces.Box(np.array([-1, -1, -1, -1, -1, -1, -1]),
                              np.array([1, 1, 1, 1, 1, 1, 1]), dtype=np.float32)
    joint_angles = np.array([0., 0., 0., 0., 0., 0., 0.])
    joint_handles = np.array([None] * 7)
    target_velocities = np.array([0., 0., 0., 0., 0., 0., 0.])
    target_pose = np.array([0.5, 0., 0.45])
    timestep = 0
    initial_policy = None

    def __init__(self, seed, rank, ep_len=64, headless=True):
        super().__init__(rank, headless)

        self.target_norm = self.normalise_coords(self.target_pose)
        self.np_random = np.random.RandomState()
        self.np_random.seed(seed + rank)
        self.ep_len = ep_len

        return_code = vrep.simxSynchronous(self.cid, enable=True)
        check_for_errors(return_code)

        return_code = vrep.simxLoadScene(self.cid, scene_path, 0, vrep.simx_opmode_blocking)
        check_for_errors(return_code)

        # Get the initial configuration of the robot (needed to later reset the robot's pose)
        self.init_config_tree, _, _, _ = self.call_lua_function('get_configuration_tree')
        _, self.init_joint_angles, _, _ = self.call_lua_function('get_joint_angles')

        for i in range(7):
            return_code, handle = vrep.simxGetObjectHandle(self.cid, 'Sawyer_joint' + str(i + 1),
                                                           vrep.simx_opmode_blocking)
            check_for_errors(return_code)
            self.joint_handles[i] = handle

        return_code, self.end_handle = vrep.simxGetObjectHandle(self.cid,
                "BaxterGripper_centerJoint", vrep.simx_opmode_blocking)
        check_for_errors(return_code)
        _, self.target_handle = vrep.simxGetObjectHandle(self.cid,
                "Waypoint", vrep.simx_opmode_blocking)
        _, self.wall_handle = vrep.simxGetObjectHandle(self.cid,
                "Wall", vrep.simx_opmode_blocking)
        self.init_wall_pos = vrep.simxGetObjectPosition(self.cid, self.wall_handle,
                -1, vrep.simx_opmode_blocking)[1]
        self.wall_distance = self.normalise_coords(self.init_wall_pos[0],
                                                   lower=0, upper=1)
        self.init_wall_rot = vrep.simxGetObjectOrientation(self.cid,
                self.wall_handle, -1, vrep.simx_opmode_blocking)[1]
        self.wall_orientation = self.init_wall_rot

        # Start the simulation (the "Play" button in V-Rep should now be in a "Pressed" state)
        return_code = vrep.simxStartSimulation(self.cid, vrep.simx_opmode_blocking)
        check_for_errors(return_code)

        self.train_initial_policy()

    # Normalise target so that x and y are in range [0, 1].
    def normalise_coords(self, coords, lower=np.array([0.2, -0.4, 0.2]),
                         upper=np.array([0.9, 0.4, 1])): # DEBUG: u/l bounds currently set for waypoint
        return (coords - lower) / (upper - lower)

    # Normalise joint angles so -pi -> 0, 0 -> 0.5 and pi -> 1. (mod pi)
    def normalise_angles(self, angles):
        js = angles / np.pi
        rem = lambda x: x - x.astype(int)
        return np.array(
            [rem((j + (np.abs(j) // 2 + 1.5) * 2) / 2.) for j in js])

    def reset(self):
        # self.target_pose[0] = self.np_random.uniform(0.125, 0.7)
        # self.target_pose[1] = self.np_random.uniform(-0.125, -0.7)
        # self.target_norm = self.normalise_target()
        self.call_lua_function('set_joint_angles', ints=self.init_config_tree, floats=self.init_joint_angles)
        # vrep.simxSetObjectPosition(self.cid, self.target_handle, -1,
        #                            self.target_pose,
        #                            vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.cid, self.wall_handle, -1,
                                   self.init_wall_pos, vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.cid, self.wall_handle, -1,
                                      self.init_wall_rot, vrep.simx_opmode_blocking)

        self.target_velocities = np.array([0., 0., 0., 0., 0., 0., 0.])
        self.joint_angles = self.init_joint_angles
        self.timestep = 0

        return self._get_obs()

    def step(self, a):
        initial_policy_input = torch.from_numpy(self.normalise_angles(self.joint_angles))
        initial_policy_action = self.initial_policy(initial_policy_input).detach().numpy()
        self.target_velocities = initial_policy_action + a  # Residual RL
        vec = self.get_end_pose() - self.target_pose
        reward_dist = - np.linalg.norm(vec)

        self.timestep += 1
        self.update_sim()

        self.wall_orientation = vrep.simxGetObjectOrientation(self.cid, self.wall_handle, -1,
                                                              vrep.simx_opmode_blocking)[1]
        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        reward_ctrl = - np.square(self.target_velocities).mean()
        reward_obstacle = - np.abs(self.wall_orientation).sum()
        reward = 0.01 * (reward_dist + reward_ctrl + reward_obstacle)

        return ob, reward, done, dict(reward_dist=reward_dist,
                                      reward_ctrl=reward_ctrl,
                                      reward_obstacle=reward_obstacle)

    def _get_obs(self):
        _, curr_joint_angles, _, _ = self.call_lua_function('get_joint_angles')
        self.joint_angles = np.array(curr_joint_angles)
        norm_joints = self.normalise_angles(self.joint_angles)

        return np.concatenate((norm_joints, self.target_norm,
                               [self.wall_distance, 0.3])) # TODO: Get height in init

    def update_sim(self):
        for handle, velocity in zip(self.joint_handles, self.target_velocities):
            return_code = vrep.simxSetJointTargetVelocity(self.cid,
                int(handle), velocity, vrep.simx_opmode_oneshot)
            check_for_errors(return_code)
        vrep.simxSynchronousTrigger(self.cid)
        vrep.simxGetPingTime(self.cid)

    def get_end_pose(self):
        pose = vrep.simxGetObjectPosition(self.cid, self.end_handle, -1,
                                          vrep.simx_opmode_blocking)[1]
        return np.array(pose)

    def render(self, mode='human'):
        pass
