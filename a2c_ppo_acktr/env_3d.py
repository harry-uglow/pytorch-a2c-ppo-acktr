# Python imports
import atexit
import platform
import signal

import numpy as np
from gym import spaces, Env
import vrep
import time
import os
from subprocess import Popen
np.set_printoptions(precision=2, linewidth=200)


# Function to check for errors when calling a remote API function
def check_for_errors(code):
    if code == vrep.simx_return_ok:
        return
    elif code == vrep.simx_return_novalue_flag:
        # Often, not really an error, so just ignore
        pass
    elif code == vrep.simx_return_timeout_flag:
        raise RuntimeError('The function timed out (probably the network is down or too slow)')
    elif code == vrep.simx_return_illegal_opmode_flag:
        raise RuntimeError('The specified operation mode is not supported for the given function')
    elif code == vrep.simx_return_remote_error_flag:
        raise RuntimeError('The function caused an error on the server side (e.g. an invalid handle was specified)')
    elif code == vrep.simx_return_split_progress_flag:
        raise RuntimeError('The communication thread is still processing previous split command of the same type')
    elif code == vrep.simx_return_local_error_flag:
        raise RuntimeError('The function caused an error on the client side')
    elif code == vrep.simx_return_initialize_error_flag:
        raise RuntimeError('A connection to vrep has not been made yet. Have you called connect()?')


# Define the port number where communication will be made to the V-Rep server
base_port_num = 19998
# Define the host where this communication is taking place (the local machine, in this case)
host = '127.0.0.1'

dir_path = os.getcwd()
scene_path = dir_path + '/reacher.ttt'

vrep_path = '/Users/Harry/Applications/V-REP_PRO_EDU_V3_6_0_Mac/vrep.app' \
            '/Contents/MacOS/vrep' \
    if platform.system() == 'Darwin' else \
    '/homes/hu115/Desktop/V-REP_PRO_EDU_V3_6_0_Ubuntu18_04/vrep.sh'


class Arm3DEnv(Env):

    observation_space = spaces.Box(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                   np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                                   dtype=np.float32)
    action_space = spaces.Box(np.array([-1, -1, -1, -1, -1, -1, -1]),
                              np.array([1, 1, 1, 1, 1, 1, 1]), dtype=np.float32)
    joint_angles = np.array([0., 0., 0., 0., 0., 0., 0.])
    joint_handles = np.array([None] * 7)
    target_velocities = np.array([0., 0., 0., 0., 0., 0., 0.])
    target_pose = np.array([0.3, -0.3, 0.025])
    # link_lengths = [0.2, 0.15, 0.1]
    timestep = 0

    def __init__(self, env_id, seed, rank, ep_len=128, headless=True):
        self.env_id = env_id
        # Launch a V-Rep server
        # Read more here: http://www.coppeliarobotics.com/helpFiles/en/commandLine.htm
        port_num = base_port_num + rank
        remote_api_string = '-gREMOTEAPISERVERSERVICE_' + str(port_num) + '_FALSE_TRUE'
        args = [vrep_path, '-h' if headless else '', remote_api_string]
        atexit.register(self.close_vrep)
        self.process = Popen(args, preexec_fn=os.setsid)
        time.sleep(6)

        self.target_norm = self.normalise_target()
        self.np_random = np.random.RandomState()
        self.np_random.seed(seed + rank)
        self.ep_len = ep_len

        self.cid = vrep.simxStart(host, port_num, True, True, 5000, 5)
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

        # Start the simulation (the "Play" button in V-Rep should now be in a "Pressed" state)
        return_code = vrep.simxStartSimulation(self.cid, vrep.simx_opmode_blocking)
        check_for_errors(return_code)

    def normalise_target(self, lower=-2.5, upper=2.5):
        return (self.target_pose - lower) / (upper - lower)

    def normalise_joints(self):  # TODO
        js = self.joint_angles / np.pi
        rem = lambda x: x - int(x)
        return np.array(
            [rem((j + (abs(j) // 2 + 1.5) * 2) / 2.) for j in js])

    def unnormalise(self, dts):  # TODO
        max_dt = np.pi / 6
        return np.array([dt * 2 * max_dt for dt in dts])

    def reset(self):
        self.call_lua_function('set_joint_angles', ints=self.init_config_tree, floats=self.init_joint_angles)

        self.target_velocities = np.array([0., 0., 0., 0., 0., 0., 0.])
        self.joint_angles = self.init_joint_angles
        self.timestep = 0

        return self._get_obs()

    def step(self, a):
        self.target_velocities = self.unnormalise(a)
        vec = self.get_end_pose() - self.target_pose
        reward_dist = - np.linalg.norm(vec)

        self.timestep += 1
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        reward_ctrl = - np.square(self.target_velocities).mean()
        reward = reward_dist + reward_ctrl

        return ob, reward, done, dict(reward_dist=reward_dist,
                                      reward_ctrl=reward_ctrl)

    def _get_obs(self):
        _, curr_joint_angles, _, _ = self.call_lua_function('get_joint_angles')
        self.joint_angles = np.array(curr_joint_angles)
        norm_joints = self.normalise_joints()
        return np.append(norm_joints, self.target_norm)

    def update_sim(self):
        # vrep.simxPauseCommunication(self.cid, True)
        for handle, velocity in zip(self.joint_handles, self.target_velocities):
            return_code = vrep.simxSetJointTargetVelocity(self.cid,
                int(handle), velocity, vrep.simx_opmode_oneshot)
            check_for_errors(return_code)
        # vrep.simxPauseCommunication(self.cid, False)
        vrep.simxSynchronousTrigger(self.cid)
        vrep.simxGetPingTime(self.cid)

    def get_end_pose(self):
        pose = vrep.simxGetObjectPosition(self.cid, self.end_handle, -1, vrep.simx_opmode_blocking)[1]
        return np.array(pose)

    def render(self, mode='human'):
        pass

    # Function to call a Lua function in V-Rep
    # Read more here: http://www.coppeliarobotics.com/helpFiles/en/remoteApiExtension.htm
    def call_lua_function(self, lua_function, ints=[], floats=[], strings=[], bytes=bytearray(),
                          opmode=vrep.simx_opmode_blocking):
        return_code, out_ints, out_floats, out_strings, out_buffer = vrep.simxCallScriptFunction(
            self.cid, 'remote_api', vrep.sim_scripttype_customizationscript, lua_function, ints,
            floats, strings, bytes, opmode)
        check_for_errors(return_code)
        return out_ints, out_floats, out_strings, out_buffer

    def close_vrep(self):
        # Shutdown
        print("Closing VREP")
        vrep.simxStopSimulation(self.cid, vrep.simx_opmode_blocking)
        vrep.simxFinish(self.cid)
        pgrp = os.getpgid(self.process.pid)
        os.killpg(pgrp, signal.SIGKILL)

    # def get_actual_velocities(self):
    #     # Get the actual velocities of the robot's joints
    #     _, actual_velocities, _, _ = self.call_lua_function('get_joint_velocities', ints=list(self.joint_handles),
    #                                                    opmode=vrep.simx_opmode_blocking)
    #     return np.array(actual_velocities)
