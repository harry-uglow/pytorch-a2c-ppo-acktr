import os

import numpy as np
from gym import spaces, Env
import vrep
from a2c_ppo_acktr.residual.initial_policy_model import InitialPolicy, train_nn

from a2c_ppo_acktr.vrep_utils import check_for_errors, VrepEnv

np.set_printoptions(precision=2, linewidth=200)

dir_path = os.getcwd()
scene_path = dir_path + '/reacher.ttt'

class Arm3DEnv(VrepEnv):

    # def generate_training_path(joint_angles, target):
    #
    #     train_path = []
    #     curr_joint_angles = np.copy(joint_angles)
    #     curr_target = compute_end_pose(curr_joint_angles)
    #
    #     for i in range(0, ep_len):
    #         if np.array_equal(target, curr_target):
    #             train_path += [np.concatenate((curr_joint_angles, target,
    #                                            [0, 0, 0]))]
    #             continue
    #         vec_to_target = target - curr_target
    #         dist_to_target = np.linalg.norm(vec_to_target)
    #
    #         if dist_to_target < dps:
    #             curr_target = target
    #         else:
    #             target_direction = vec_to_target / dist_to_target
    #             curr_target = curr_target + (target_direction * dps)
    #
    #         d_theta = compute_d_theta(curr_joint_angles, curr_target)
    #
    #         train_path += [np.concatenate((curr_joint_angles, target, d_theta))]
    #
    #         curr_joint_angles[0] += d_theta[0]
    #         curr_joint_angles[1] += d_theta[1]
    #         curr_joint_angles[2] += d_theta[2]
    #
    #     return np.array(train_path)

    def get_demo_path(self):
        num_path_points = 17
        _, path, _, _ = self.call_lua_function('solve_ik')
        path = np.reshape(path, (num_path_points, len(self.joint_handles)))
        distances = np.array([path[i + 1] - path[i]
                              for i in range(0, len(path) - 1)])
        velocities = distances * 20 # Distances should be covered in 0.05s
        return path[:-1], velocities

    def train_initial_policy(self):
        path_poses, train_y = self.get_demo_path()
        train_x = self.normalise_joints(path_poses)
        num_joints = len(self.joint_handles)
        net = train_nn(InitialPolicy(num_joints, num_joints), train_x, train_y)
        return net

    observation_space = spaces.Box(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                   np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                                   dtype=np.float32)
    action_space = spaces.Box(np.array([-1, -1, -1, -1, -1, -1, -1]),
                              np.array([1, 1, 1, 1, 1, 1, 1]), dtype=np.float32)
    joint_angles = np.array([0., 0., 0., 0., 0., 0., 0.])
    joint_handles = np.array([None] * 7)
    target_velocities = np.array([0., 0., 0., 0., 0., 0., 0.])
    target_pose = np.array([0.3, -0.3, 0.025])
    timestep = 0

    def __init__(self, seed, rank, ep_len=64, headless=True):
        super().__init__(rank, headless)

        self.target_norm = self.normalise_target()
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
                "Sawyer_tip", vrep.simx_opmode_blocking)
        check_for_errors(return_code)
        _, self.target_handle = vrep.simxGetObjectHandle(self.cid,
                "Target", vrep.simx_opmode_blocking)

        # self.train_initial_policy() # TODO

        # Start the simulation (the "Play" button in V-Rep should now be in a "Pressed" state)
        return_code = vrep.simxStartSimulation(self.cid, vrep.simx_opmode_blocking)
        check_for_errors(return_code)

    # Normalise target so that x and y are in range [0, 1].
    def normalise_target(self, lower=0.125, upper=0.7):
        return (np.abs(self.target_pose) - lower) / (upper - lower)

    # Normalise joint angles so -pi -> 0, 0 -> 0.5 and pi -> 1. (mod pi)
    def normalise_joints(self, joint_angles):
        js = joint_angles / np.pi
        rem = lambda x: x - x.astype(int)
        return np.array(
            [rem((j + (np.abs(j) // 2 + 1.5) * 2) / 2.) for j in js])

    # TODO: Remove
    def unnormalise(self, dts):
        max_dt = np.pi / 6
        return np.array([dt * 2 * max_dt for dt in dts])

    def reset(self):
        self.target_pose[0] = self.np_random.uniform(0.125, 0.7)
        self.target_pose[1] = self.np_random.uniform(-0.125, -0.7)
        self.target_norm = self.normalise_target()
        self.call_lua_function('set_joint_angles', ints=self.init_config_tree, floats=self.init_joint_angles)
        vrep.simxSetObjectPosition(self.cid, self.target_handle, -1,
                                   self.target_pose,
                                   vrep.simx_opmode_blocking)

        self.target_velocities = np.array([0., 0., 0., 0., 0., 0., 0.])
        self.joint_angles = self.init_joint_angles
        self.timestep = 0

        return self._get_obs()

    def step(self, a):
        self.target_velocities = self.unnormalise(a)
        vec = self.get_end_pose() - self.target_pose
        reward_dist = - np.linalg.norm(vec) / 100.

        self.timestep += 1
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        reward_ctrl = - np.square(self.target_velocities).mean() / 100.
        reward = reward_dist + reward_ctrl

        return ob, reward, done, dict(reward_dist=reward_dist,
                                      reward_ctrl=reward_ctrl)

    def _get_obs(self):
        _, curr_joint_angles, _, _ = self.call_lua_function('get_joint_angles')
        self.joint_angles = np.array(curr_joint_angles)
        norm_joints = self.normalise_joints(np.array(curr_joint_angles))
        return np.append(norm_joints, self.target_norm)

    def update_sim(self):
        for handle, velocity in zip(self.joint_handles, self.target_velocities):
            return_code = vrep.simxSetJointTargetVelocity(self.cid,
                int(handle), velocity, vrep.simx_opmode_oneshot)
            check_for_errors(return_code)
        vrep.simxSynchronousTrigger(self.cid)
        vrep.simxGetPingTime(self.cid)

    def get_end_pose(self):
        pose = vrep.simxGetObjectPosition(self.cid, self.end_handle, -1, vrep.simx_opmode_blocking)[1]
        return np.array(pose)

    def render(self, mode='human'):
        pass

    ##### RESIDUAL RL #####
    # ip_action = initial_policy.act(obs)
    # complete_action =
    #######################
