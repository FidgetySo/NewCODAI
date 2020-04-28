from threading import Thread

import numpy as np
import keras.backend.tensorflow_backend as backend
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2

from ddqn import DDQN

from mss import mss

import skimage.io
import skimage.util
import skimage.morphology
import skimage.segmentation
import skimage.measure

import pyautogui
pyautogui.FAILSAFE = False

import keyboard
import ctypes
import pynput
from pynput.keyboard import Key,Listener,KeyCode
from pynput import keyboard

SendInput=ctypes.windll.user32.SendInput
PUL=ctypes.POINTER(ctypes.c_ulong)
paused = False
def on_release(key):
    print('{0} released'.format(
        key))
    if key == keyboard.Key.esc:
        global paused
        paused = True

listener = keyboard.Listener(on_release=on_release)
listener.start()
class KeyBdInput(ctypes.Structure):
    _fields_=[("wVk",ctypes.c_ushort),
              ("wScan",ctypes.c_ushort),
              ("dwFlags",ctypes.c_ulong),
              ("time",ctypes.c_ulong),
              ("dwExtraInfo",PUL)]


class HardwareInput(ctypes.Structure):
    _fields_=[("uMsg",ctypes.c_ulong),
              ("wParamL",ctypes.c_short),
              ("wParamH",ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_=[("dx",ctypes.c_long),
              ("dy",ctypes.c_long),
              ("mouseData",ctypes.c_ulong),
              ("dwFlags",ctypes.c_ulong),
              ("time",ctypes.c_ulong),
              ("dwExtraInfo",PUL)]


class Input_I(ctypes.Union):
    _fields_=[("ki",KeyBdInput),
              ("mi",MouseInput),
              ("hi",HardwareInput)]


class Input(ctypes.Structure):
    _fields_=[("type",ctypes.c_ulong),
              ("ii",Input_I)]


def set_pos(x,y):
    x=1+int(x*65536./1920.)
    y=1+int(y*65536./1080.)
    extra=ctypes.c_ulong(0)
    ii_=pynput._util.win32.INPUT_union()
    ii_.mi=pynput._util.win32.MOUSEINPUT(x,y,0,(0x0001|0x8000),0,ctypes.cast(ctypes.pointer(extra),ctypes.c_void_p))
    command=pynput._util.win32.INPUT(ctypes.c_ulong(0),ii_)
    SendInput(1,ctypes.pointer(command),ctypes.sizeof(command))


def left_click():
    extra=ctypes.c_ulong(0)
    ii_=pynput._util.win32.INPUT_union()
    ii_.mi=pynput._util.win32.MOUSEINPUT(0,0,0,0x0002,0,ctypes.cast(ctypes.pointer(extra),ctypes.c_void_p))
    x=pynput._util.win32.INPUT(ctypes.c_ulong(0),ii_)
    SendInput(1,ctypes.pointer(x),ctypes.sizeof(x))

    extra=ctypes.c_ulong(0)
    ii_=pynput._util.win32.INPUT_union()
    ii_.mi=pynput._util.win32.MOUSEINPUT(0,0,0,0x0004,0,ctypes.cast(ctypes.pointer(extra),ctypes.c_void_p))
    x=pynput._util.win32.INPUT(ctypes.c_ulong(0),ii_)
    SendInput(1,ctypes.pointer(x),ctypes.sizeof(x))


def hold_left_click():
    extra=ctypes.c_ulong(0)
    ii_=pynput._util.win32.INPUT_union()
    ii_.mi=pynput._util.win32.MOUSEINPUT(0,0,0,0x0002,0,ctypes.cast(ctypes.pointer(extra),ctypes.c_void_p))
    x=pynput._util.win32.INPUT(ctypes.c_ulong(0),ii_)
    SendInput(1,ctypes.pointer(x),ctypes.sizeof(x))


def release_left_click():
    extra=ctypes.c_ulong(0)
    ii_=pynput._util.win32.INPUT_union()
    ii_.mi=pynput._util.win32.MOUSEINPUT(0,0,0,0x0004,0,ctypes.cast(ctypes.pointer(extra),ctypes.c_void_p))
    x=pynput._util.win32.INPUT(ctypes.c_ulong(0),ii_)
    SendInput(1,ctypes.pointer(x),ctypes.sizeof(x))


def right_click():
    extra=ctypes.c_ulong(0)
    ii_=pynput._util.win32.INPUT_union()
    ii_.mi=pynput._util.win32.MOUSEINPUT(0,0,0,0x0008,0,ctypes.cast(ctypes.pointer(extra),ctypes.c_void_p))
    x=pynput._util.win32.INPUT(ctypes.c_ulong(0),ii_)
    SendInput(1,ctypes.pointer(x),ctypes.sizeof(x))

    extra=ctypes.c_ulong(0)
    ii_=pynput._util.win32.INPUT_union()
    ii_.mi=pynput._util.win32.MOUSEINPUT(0,0,0,0x0010,0,ctypes.cast(ctypes.pointer(extra),ctypes.c_void_p))
    x=pynput._util.win32.INPUT(ctypes.c_ulong(0),ii_)
    SendInput(1,ctypes.pointer(x),ctypes.sizeof(x))


def hold_right_click():
    extra=ctypes.c_ulong(0)
    ii_=pynput._util.win32.INPUT_union()
    ii_.mi=pynput._util.win32.MOUSEINPUT(0,0,0,0x0008,0,ctypes.cast(ctypes.pointer(extra),ctypes.c_void_p))
    x=pynput._util.win32.INPUT(ctypes.c_ulong(0),ii_)
    SendInput(1,ctypes.pointer(x),ctypes.sizeof(x))


def release_right_click():
    extra=ctypes.c_ulong(0)
    ii_=pynput._util.win32.INPUT_union()
    ii_.mi=pynput._util.win32.MOUSEINPUT(0,0,0,0x0010,0,ctypes.cast(ctypes.pointer(extra),ctypes.c_void_p))
    x=pynput._util.win32.INPUT(ctypes.c_ulong(0),ii_)
    SendInput(1,ctypes.pointer(x),ctypes.sizeof(x))


def Press_Key(hexKeyCode):
    extra=ctypes.c_ulong(0)
    ii_=pynput._util.win32.INPUT_union()
    ii_.ki=pynput._util.win32.KEYBDINPUT(0,hexKeyCode,0x0008,0,ctypes.cast(ctypes.pointer(extra),ctypes.c_void_p))
    x=pynput._util.win32.INPUT(ctypes.c_ulong(1),ii_)
    SendInput(1,ctypes.pointer(x),ctypes.sizeof(x))

    extra=ctypes.c_ulong(0)
    ii_=pynput._util.win32.INPUT_union()
    ii_.ki=pynput._util.win32.KEYBDINPUT(0,hexKeyCode,0x0008|0x0002,0,
                                         ctypes.cast(ctypes.pointer(extra),ctypes.c_void_p))
    x=pynput._util.win32.INPUT(ctypes.c_ulong(1),ii_)
    SendInput(1,ctypes.pointer(x),ctypes.sizeof(x))


def HoldKey(hexKeyCode):
    extra=ctypes.c_ulong(0)
    ii_=pynput._util.win32.INPUT_union()
    ii_.ki=pynput._util.win32.KEYBDINPUT(0,hexKeyCode,0x0008,0,ctypes.cast(ctypes.pointer(extra),ctypes.c_void_p))
    x=pynput._util.win32.INPUT(ctypes.c_ulong(1),ii_)
    SendInput(1,ctypes.pointer(x),ctypes.sizeof(x))


def ReleaseKey(hexKeyCode):
    extra=ctypes.c_ulong(0)
    ii_=pynput._util.win32.INPUT_union()
    ii_.ki=pynput._util.win32.KEYBDINPUT(0,hexKeyCode,0x0008|0x0002,0,
                                         ctypes.cast(ctypes.pointer(extra),ctypes.c_void_p))
    x=pynput._util.win32.INPUT(ctypes.c_ulong(1),ii_)
    SendInput(1,ctypes.pointer(x),ctypes.sizeof(x))

DISCOUNT = 0.99
H_SIZE = 512
REPLAY_MEMORY_SIZE = 10_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'COD'
MODEL_FILE = 'models/MODEL.h5'
MIN_REWARD = -2000  # For model save
MEMORY_FRACTION = 0.20

EPOCHS=4

# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 5  # episodes

mouse1 = False
mouse2 = False
mouse3 = False
mouse4 = False
def start_mouse():
    global mouse1
    global mouse2
    global mouse3
    global mouse4
    while True:
        if mouse1:
            set_pos(965,540)
        elif mouse2:
            set_pos(960,545)
        elif mouse3:
            set_pos(955,540)
        elif mouse4:
            set_pos(960,535)
        else:
            mouse1=False
            mouse2=False
            mouse3=False
            mouse4=False

worker = Thread(target=start_mouse, args=())
worker.setDaemon(True)
worker.start()

class Blob:
    def action(self, choice):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        '''
        global mouse1
        global mouse2
        global mouse3
        global mouse4
        if choice == 0:
            pyautogui.mouseDown(button="left")
        elif choice == 1:
            pyautogui.mouseDown(button="right")
        elif choice == 2:
            pyautogui.mouseUp(button="left")
        elif choice == 3:
            pyautogui.mouseUp(button="right")
        elif choice == 4:
            mouse1 = True
            mouse2 = False
            mouse3=False
            mouse4=False
        elif choice == 5:
            mouse2 = True
            mouse1=False
            mouse3=False
            mouse4=False
        elif choice == 6:
            mouse3 = True
            mouse1=False
            mouse2=False
            mouse4=False
        elif choice == 7:
            mouse4 = True
            mouse1=False
            mouse2=False
            mouse3=False
        elif choice == 8:
            mouse4=False
            mouse1=False
            mouse2=False
            mouse3=False
        elif choice == 9:
            HoldKey(0x11)
            HoldKey(0x2A)
        elif choice == 10:
            ReleaseKey(0x11)
            ReleaseKey(0x2A)




class BlobEnv:
    SIZE = 75
    image = Image.open('ref.png')
    # convert image to numpy array
    ref = np.asarray(image)
    ref=cv2.cvtColor(ref, cv2.COLOR_BGRA2GRAY)
    player = Blob()
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    ACTION_SPACE_SIZE = 10

    def reset(self):
        self.episode_step = 0
        observation = np.array(self.get_image())
        observation=cv2.resize(observation,(self.SIZE,self.SIZE),interpolation=cv2.INTER_AREA)
        observation=observation[...,:3]
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        img = np.array(self.get_image())
        new_observation=cv2.resize(img,(self.SIZE,self.SIZE),interpolation=cv2.INTER_AREA)
        new_observation=new_observation[...,:3]

        done = False
        done_check = self.done_check(img, self.ref)
        reward_xp = self.get_xp(img)

        reward_health =self.get_health(img)

        if reward_xp > 0:
            reward = 160
        elif reward_health > 0:
            reward = -85
        else:
            reward = 0
        if done_check > 0.5:
            done = True
            Press_Key(0x21)
            ReleaseKey(0x1F)
            ReleaseKey(0x11)
            ReleaseKey(0x2A)
            pyautogui.mouseUp(button="right")
            pyautogui.mouseUp(button="left")
        return new_observation, reward, done

    # FOR CNN #
    def get_image(self):
        with mss() as sct:
            img = np.array(sct.grab(sct.monitors[1]))
        return img
    def done_check(self, img, ref):
        img=np.array(img)[...,:3]
        img=img[60:60+52,57:57+211,:]
        game_frame = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        ssim_score=skimage.measure.compare_ssim(
            game_frame,
            ref ,
            multichannel=True
        )
        return ssim_score
    def get_xp(self, image_xp):
        image=image_xp[ 407:407 + 215 , 980:980 + 323 , : ]
        image=cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        yellow_min=np.array([255,194,21] , np.uint8)
        yellow_max=np.array([255,194,21] , np.uint8)

        dst=cv2.inRange(image , yellow_min , yellow_max)
        no_yellow=cv2.countNonZero(dst)
        print('The number of yellow  pixels are: ' + str(no_yellow))
        return no_yellow
    def get_health(self, image):
        image=cv2.cvtColor(image , cv2.COLOR_BGRA2RGB)
        red_min=np.array([ 117, 54, 34] , np.uint8)
        red_max=np.array([ 117, 54, 34 ] , np.uint8)

        dst=cv2.inRange(image , red_min , red_max)
        no_red =cv2.countNonZero(dst)
        print('The number of red pixels are: ' + str(no_red))
        return no_red
env = BlobEnv()

# For stats
ep_rewards = [0]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_memory_growth(physical_devices[1], True)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, MODEL_NAME)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


agent = DDQN(10,(env.OBSERVATION_SPACE_VALUES))
agent.load_weights(MODEL_FILE)
tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False

    while not done:
        if paused:
            time.sleep(15)
            paused = False
        start_time=time.time()
        # This part stays mostly the same, the change is to query a model for Q values
        action = agent.policy_action(current_state)

        new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        # Every step we update replay memory and train main network
        agent.memorize((current_state, action, reward, new_state, done))
        agent.train(current_state, new_state, done, step)

        current_state = new_state
        step += 1
        print(step)
        print("FPS: ",1.0/(time.time()-start_time))

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.save_weights(MODEL_FILE)

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)