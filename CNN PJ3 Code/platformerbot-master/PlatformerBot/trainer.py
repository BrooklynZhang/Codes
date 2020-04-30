#PlatformBotProj
import numpy as np
import gym
import retro
from PIL import Image
from io import BytesIO
import os
import datetime
import threading
import time
import random

from .replay import Replay, ReplayMemory


class TrainerThread(threading.Thread):
    def __init__(self, bot=None, **kwargs):
        super(TrainerThread, self).__init__(**kwargs)
        self.stop_flag = False
        self.bot = bot
        self.mode = ""
        self.replay_memory = None

    def run(self):
        sleep = False
        while not self.stop_flag:
            if sleep:
                time.sleep(1)
            sleep = True

            if self.mode == "replay":
                if self.replay_memory is None:
                    continue

                batch = self.replay_memory.sample(64)
                result = self.bot.train_value_network_batch_step(batch)
                value_loss = result["loss"]
                result = self.bot.train_policy_network_batch_step(batch)
                policy_loss = result["loss"]
                if random.randint(0, 100) == 0:
                    self.bot.normalize_weight()
                    self.bot.save_model()
                print("Value loss:", value_loss)
                print("Policy loss:", policy_loss)
            else:
                continue

            sleep = False

    def train_with_replay(self, replay_memory):
        self.replay_memory = replay_memory
        self.mode = "replay"

    def pause_training(self):
        self.mode = ""
        self.bot.save_model()

    def stop(self):
        self.stop_flag = True


class Trainer:
    def __init__(self, bot=None):
        self.bot = bot
        self.thread = TrainerThread(bot=bot)
        self.thread.start()

    def train_with_replays(self, replays):
        replay_memory = ReplayMemory.from_replays(replays)
        self.thread.train_with_replay(replay_memory)

    def pause(self):
        self.thread.pause_training()

    def __del__(self):
        if self.thread.isAlive():
            self.thread.stop()
            self.thread.join()
