# Python imports
import numpy as np
from gym import spaces, Env
import vrep
import time
import os
from subprocess import Popen

port_num = 19996
host = '127.0.0.1'
# Launch V-Rep in headless mode
remote_api_string = '-gREMOTEAPISERVERSERVICE_' + str(port_num) + '_FALSE_TRUE'
args = ['/Users/Harry/Applications/V-REP_PRO_EDU_V3_6_0_Mac/vrep.app/Contents'
        '/MacOS/vrep',
        '-h',
        remote_api_string]


class Arm3DEnv(Env):

    observation_space = spaces.Box(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                   np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                                   dtype=np.float32)
    action_space = spaces.Box(np.array([-1, -1, -1, -1, -1, -1, -1]),
                              np.array([1, 1, 1, 1, 1, 1, 1]), dtype=np.float32)
    joint_angles = np.array([0., 0., 0., 0., 0., 0., 0.])
    joint_handles = np.array([0, 0, 0, 0, 0, 0, 0])
    target_velocities = np.array([0., 0., 0., 0., 0., 0., 0.])
    target_pose = np.array([0.625, -0.4, 0.025])
    # link_lengths = [0.2, 0.15, 0.1]
    timestep = 0

    def __init__(self, seed, ep_len=128):
        self.process = Popen(args, preexec_fn=os.setsid)
        time.sleep(6)

        self.target_norm = self.normalise_target()
        self.np_random = np.random.RandomState()
        self.np_random.seed(seed)
        self.ep_len = ep_len

        self.cid = vrep.simxStart(host, port_num, True, True, 5000, 5)
        print(self.cid)
        vrep.simxSynchronous(self.cid, enable=True)
        cd = vrep.simxLoadScene(self.cid, '/Users/Harry/Uni/Project/scribbles/'
                                     'pytorch-a2c-ppo-acktr/reacher.ttt',
                                0, vrep.simx_opmode_blocking)
        print(cd)

        for i in range(7):
            code = -1
            cnt = 0
            while code != 0:
                code, self.joint_handles[i] = np.array(vrep.simxGetObjectHandle(
                    self.cid, "Sawyer_joint" + str(i + 1),
                    vrep.simx_opmode_blocking))
                cnt += 1
                if cnt == 5 and code != 0:
                    raise ConnectionError("Could not find joint " + str(i + 1))

        code = -1
        cnt = 0
        while code != 0:
            code, self.end_handle = vrep.simxGetObjectHandle(self.cid,
                     "BaxterGripper_centerJoint", vrep.simx_opmode_blocking)
            cnt += 1
            if cnt == 5 and code != 0:
                raise ConnectionError("Could not find end joint")

        self.joint_angles = np.array([vrep.simxGetJointPosition(
            self.cid, int(handle), vrep.simx_opmode_streaming)[1]
                                      for handle in self.joint_handles])
        vrep.simxGetJointPosition(self.cid, self.end_handle,
                                  vrep.simx_opmode_streaming)

        print("Environment is loaded: ", code, self.joint_handles,
              self.end_handle)
        self.start_time = time.time()

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
        curr_time = time.time()
        print("Episode duration: ", curr_time - self.start_time)
        self.start_time = curr_time
        vrep.simxStopSimulation(self.cid, vrep.simx_opmode_oneshot)

        # print("Environment is reset: ", self.joint_handles, self.end_handle,
        #       self.joint_angles)

        vrep.simxPauseCommunication(self.cid, True)
        for handle in self.joint_handles:
            vrep.simxSetJointTargetVelocity(self.cid,
                int(handle), 0, vrep.simx_opmode_oneshot)

        self.joint_angles = np.array([0., 0., 0., 0., 0., 0., 0.])
        self.timestep = 0
        vrep.simxPauseCommunication(self.cid, False)

        vrep.simxStartSimulation(self.cid, vrep.simx_opmode_oneshot)
        return self._get_obs()

    def step(self, a):
        # start_time = time.time()
        joint_velocities = self.unnormalise(a)
        vec = self.get_end_pose() - self.target_pose
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(joint_velocities).mean()
        reward = reward_dist + reward_ctrl

        self.target_velocities = joint_velocities
        self.timestep += 1
        self.update_sim()

        ob = self._get_obs()
        done = (self.timestep == self.ep_len)

        # print("Step duration: ", time.time() - start_time)
        return ob, reward, done, dict(reward_dist=reward_dist,
                                      reward_ctrl=reward_ctrl)

    def _get_obs(self):
        try:
            self.joint_angles = np.array([vrep.simxGetJointPosition(self.cid,
                int(handle), vrep.simx_opmode_buffer)[1]
                                        for handle in self.joint_handles])
        except ValueError:
            print("NPE")
        self.joint_angles[1] = self.joint_angles[1] + np.pi / 2
        norm_joints = self.normalise_joints()
        return np.append(norm_joints, self.target_norm)

    def update_sim(self):
        vrep.simxPauseCommunication(self.cid, True)
        for handle, velocity in zip(self.joint_handles, self.target_velocities):
            vrep.simxSetJointTargetVelocity(self.cid,
                int(handle), velocity, vrep.simx_opmode_oneshot)
        vrep.simxPauseCommunication(self.cid, False)
        vrep.simxSynchronousTrigger(self.cid)

    def get_end_pose(self):
        return vrep.simxGetJointPosition(self.cid,
            self.end_handle, vrep.simx_opmode_buffer)[1]

    def render(self, mode='human'):
        pass
