import atexit
import os
import platform
import signal
import time

from gym import Env
from subprocess import Popen
import vrep


# Function to check for errors when calling a remote API function
def check_for_errors(code):
    if code == vrep.simx_return_ok:
        return
    elif code == vrep.simx_return_timeout_flag:
        raise RuntimeError('The function timed out (probably the network is down or too slow)')
    elif code == vrep.simx_return_novalue_flag:
        # Often, not really an error, so just ignore
        pass
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
# Define the host where this communication is taking place
host = '127.0.0.1'

vrep_path = '/Users/Harry/Applications/V-REP_PRO_EDU_V3_6_1_Mac/vrep.app' \
            '/Contents/MacOS/vrep' \
    if platform.system() == 'Darwin' else \
    '/homes/hu115/Desktop/V-REP_PRO_EDU_V3_6_1_Ubuntu18_04/vrep.sh'


class VrepEnv(Env):
    """
    TODO: Document
    A goal-based environment. It functions just as any regular OpenAI Gym environment but it
    imposes a required structure on the observation_space. More concretely, the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    """

    def __init__(self, rank, headless):
        # Launch a V-Rep server
        # Read more here: http://www.coppeliarobotics.com/helpFiles/en/commandLine.htm
        port_num = base_port_num + rank
        if not headless:  # DEBUG: Helps run enjoy while Train is running
            port_num += 16
        remote_api_string = '-gREMOTEAPISERVERSERVICE_' + str(port_num) + '_FALSE_TRUE'
        args = [vrep_path, '-h' if headless else '', remote_api_string]
        atexit.register(self.close)
        self.process = Popen(args, preexec_fn=os.setsid)
        time.sleep(6)

        self.cid = vrep.simxStart(host, port_num, True, True, 5000, 5)

    # Function to call a Lua function in V-Rep
    # Read more here: http://www.coppeliarobotics.com/helpFiles/en/remoteApiExtension.htm
    def call_lua_function(self, lua_function, ints=[], floats=[], strings=[],
                          bytes=bytearray(),
                          opmode=vrep.simx_opmode_blocking):
        return_code, out_ints, out_floats, out_strings, out_buffer = vrep.simxCallScriptFunction(
            self.cid, 'remote_api', vrep.sim_scripttype_customizationscript,
            lua_function, ints,
            floats, strings, bytes, opmode)
        check_for_errors(return_code)
        return out_ints, out_floats, out_strings, out_buffer

    def close(self):
        # Shutdown
        print("Closing VREP")
        vrep.simxStopSimulation(self.cid, vrep.simx_opmode_blocking)
        vrep.simxFinish(self.cid)
        try:
            pgrp = os.getpgid(self.process.pid)
            os.killpg(pgrp, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def render(self, mode='human'):
        pass
