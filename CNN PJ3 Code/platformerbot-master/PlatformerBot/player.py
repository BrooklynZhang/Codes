#PlatformBotProj
import numpy as np
import kivy
from kivy.app import App
from kivy.properties import Property, BooleanProperty, NumericProperty, StringProperty
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
from kivy.clock import Clock
import kivy.uix.image
import gym
import retro
from PIL import Image
from io import BytesIO
import os
import datetime
import random
from .replay import Replay


class RetroPlayer(BoxLayout):
    mode = StringProperty("interactive")
    fps = NumericProperty(30)
    paused = BooleanProperty(False)
    recording = BooleanProperty(False)
    current_replay_frame = NumericProperty(0)

    def __init__(self, bot, **kwargs):
        super(RetroPlayer, self).__init__(**kwargs)

        self.retro_canvas = kivy.uix.image.Image(allow_stretch=True)

        self.add_widget(self.retro_canvas)

        self.keyboard = Window.request_keyboard(self.on_keyboard_closed, self)
        self.keyboard.bind(on_key_down=self.on_key_down, on_key_up=self.on_key_up)

        self.input_size = 22
        self.input_map = {
            "w": 4,
            "s": 5,
            "a": 6,
            "d": 7,
            "j": 8,
            "k": 9,
        }
        self.input_state = [0 for i in range(self.input_size)]

        self.bot = bot
        self.retro_game = None
        self.retro_env = None
        self.retro_observation = None
        self.retro_done = True
        self.retro_action_size = 0
        self.train_random_chance = 0.01
        self.reward_sum = 0
        self.take_control = False

        self.recreate_retro_env(game="SuperMarioBros-Nes")
        #self.recreate_retro_env(game="Airstriker-Genesis")

        self.replay = None

        self.update_schedule = Clock.schedule_interval(self.update, 1.0 / self.fps)

    def on_keyboard_closed(self):
        self.keyboard.unbind(on_key_down=self.on_key_down, on_key_up=self.on_key_up)

    def on_key_down(self, keyboard, keycode, text, modifiers):
        key_name = keycode[1]
        if key_name in self.input_map.keys():
            self.input_state[self.input_map[key_name]] = 1
        if key_name == "shift":
            self.take_control = True
        return True

    def on_key_up(self, keyboard, keycode):
        key_name = keycode[1]
        if key_name in self.input_map.keys():
            self.input_state[self.input_map[key_name]] = 0
        if key_name == "shift":
            self.take_control = False
        return True

    def update(self, dt):
        if self.mode == "interactive":
            # Play the game interactively
            if self.paused:
                return
            action = np.asarray(self.input_state)[:self.retro_action_size]
            observation, reward, done, info = self.retro_env.step(action)
            self.reward_sum += reward
            if self.recording and self.replay is not None:
                self.replay.append(action, observation, reward, done, info)
            self.retro_observation = observation
            self.retro_done = done
            self.update_canvas()
            if self.retro_done:
                if self.recording:
                    self.record_stop()
                    self.record_start()
                self.retro_observation = self.retro_env.reset()
                self.reward_sum = 0
        elif self.mode == "playback":
            # show next frame of the replay
            if self.paused:
                return
            if self.replay is not None:
                if self.current_replay_frame < len(self.replay):
                    action, observation, reward, done, info = self.replay[self.current_replay_frame]
                    self.retro_observation = observation
                    self.update_canvas()
                    self.current_replay_frame += 1

        elif self.mode == "train":
            observation = self.retro_observation
            policy_tensor, value_tensor = self.bot.forward(observation)
            policy_tensor = policy_tensor.view(-1)

            if self.take_control:
                action = np.asarray(self.input_state)[:self.retro_action_size]
            elif np.random.uniform() < self.train_random_chance:
                action = np.asarray(self.retro_env.action_space.sample())
            else:
                action_size = self.retro_env.action_space.n
                action = policy_tensor.detach().cpu().numpy()[:action_size]
                action_rd = np.random.random_sample(action.shape)
                action = (action > action_rd).astype(np.int)

            next_observation, reward, done, info = self.retro_env.step(action)
            self.reward_sum += reward
            self.bot.train_step(action, policy_tensor, value_tensor, next_observation, reward)

            if random.randint(0, 2000) == 0:
                self.bot.save_model()
            if self.recording and self.replay is not None:
                self.replay.append(action, observation, reward, done, info)
            self.retro_done = done
            self.retro_observation = next_observation
            self.update_canvas()
            if self.retro_done:
                if self.recording:
                    self.record_stop()
                    self.record_start()
                self.retro_observation = self.retro_env.reset()
                self.reward_sum = 0

        elif self.mode == "demo":
            observation = self.retro_observation

            action, value = self.bot.forward(observation)
            action = action.detach().cpu().numpy()
            value = value.detach().cpu().numpy()

            if self.take_control:
                action = np.asarray(self.input_state)[:self.retro_action_size]

            action = action.reshape(-1)[:self.retro_action_size]
            action_rd = np.random.random_sample(action.shape)
            action = (action > action_rd).astype(np.int)
            observation, reward, done, info = self.retro_env.step(action)
            self.reward_sum += reward
            if self.recording and self.replay is not None:
                self.replay.append(action, observation, reward, done, info)
            self.retro_observation = observation
            self.retro_done = done
            self.update_canvas()
            if self.retro_done:
                if self.recording:
                    self.record_stop()
                    self.record_start()
                self.retro_observation = self.retro_env.reset()
                self.reward_sum = 0

    def update_canvas(self):
        frame_pil = Image.fromarray(np.asarray(self.retro_observation))
        buf = BytesIO()
        frame_pil.save(buf, format="png")
        buf.seek(0)
        frame_kv = kivy.core.image.Image(BytesIO(buf.read()), ext="png")
        self.retro_canvas.texture = frame_kv.texture

    def recreate_retro_env(self, game=None):
        if self.retro_env is not None:
            self.retro_env.close()
        self.retro_env = retro.make(game=game)
        self.retro_game = game
        self.retro_action_size = self.retro_env.action_space.n
        self.reset_retro_env()

    def reset_retro_env(self):
        if self.retro_env:
            self.retro_observation = self.retro_env.reset()
            self.retro_done = False

    def record_start(self):
        if not self.recording:
            self.replay = Replay()
            self.current_replay_frame = 0
            self.recording = True

    def record_stop(self):
        if self.recording:
            self.record_save()
        self.recording = False

    def record_save(self, filename=None, path="replays"):
        if filename is None:
            date = datetime.datetime.now()
            filename = self.retro_game + " " + date.strftime("%Y-%m-%d %H-%M-%S") + ".pbreplay"

        if not os.path.exists(path):
            os.makedirs(path)

        fullpath = os.path.join(path, filename)
        with open(fullpath, mode="wb") as file:
            self.replay.dump(file)
        return fullpath

    def change_mode(self, mode):
        # cleanup
        if self.mode == "interactive":
            if self.recording:
                self.record_stop()

        supported_modes = ["interactive", "playback", "train", "demo"]
        if mode in supported_modes:
            self.mode = mode
        else:
            raise RuntimeError("Unsupported mode "+mode)

    def load_replay(self, replay):
        self.replay = replay

    def playback_pause(self):
        self.paused = True

    def playback_unpause(self):
        self.paused = False

    def playback_seek(self, position):
        self.current_replay_frame = position

    def playback_len(self):
        if self.replay:
            return len(self.replay)
        return 0

    def env_step(self, action):
        # Wrapper for direct interact with the env
        return self.retro_env.step(action)

    def env_reset(self):
        # Wrapper for direct interact with the env
        return self.retro_env.reset()
