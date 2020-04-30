#PlatformBotProj
from . import model
from . import replay
import numpy as np
import torch
import torchvision
import gym
import PIL
import retro
import os
import threading
import itertools
from .replay import ReplayMemory


class PlatformBot:
    def __init__(self):
        self.device = "cuda"
        self.save_path = "saves1/"
        self.common_cnn = model.CommonCNN().to(self.device)
        self.policy_model = model.PolicyNetwork(common_cnn=self.common_cnn).to(self.device)
        self.value_model = model.ValueNetwork(common_cnn=self.common_cnn).to(self.device)
        self.model_input_size = (320, 240)

        self.discount_factor = 0.99
        self.random_chance = 0.01
        self.key_press_penalty = 0.01
        self.step_penalty = 0.01
        lr = 1e-1

        self.optimizer = torch.optim.Adam(list(self.policy_model.parameters())+list(self.value_model.parameters()), lr=lr)

        try:
            self.load_model()
        except Exception as e:
            print("Failed to load model: ")
            print(e)

    def forward_detailed(self, observation):
        observation = self.preprocess_input_image(observation)
        observation_tensor = torch.tensor(observation, dtype=torch.float, device=self.device).reshape((-1, *observation.shape))
        policy_data = self.policy_model.forward_detailed(observation_tensor)
        value_date = self.value_model.forward_detailed(observation_tensor)
        return {
            "common-cnn": policy_data["common-cnn"],
            "policy": policy_data["policy"],
            "value": value_date["value"],
        }

    def forward(self, observation):
        observation = self.preprocess_input_image(observation)
        observation_tensor = torch.tensor(observation, dtype=torch.float, device=self.device).reshape((-1, *observation.shape))
        policy = self.policy_model.forward(observation_tensor)
        value = self.value_model.forward(observation_tensor)
        return policy, value

    def train_value_network_batch_step(self, batch):
        observations, actions, rewards, dones, next_observations = batch
        observations = self.preprocess_input_image(observations)
        next_observations = self.preprocess_input_image(next_observations)

        observations_tensor = torch.tensor(observations, dtype=torch.float, device=self.device)
        next_observations_tensor = torch.tensor(next_observations, dtype=torch.float, device=self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float, device=self.device).reshape(rewards.shape[0], 1)

        next_value = self.value_model(next_observations_tensor)
        target_value = reward_tensor + self.discount_factor * next_value
        predict_value = self.value_model(observations_tensor)
        advance = target_value - predict_value

        value_loss = (target_value - predict_value).pow(2).mean()
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

        return {
            "loss": value_loss.detach().cpu().numpy(),
        }

    def train_policy_network_batch_step(self, batch):
        observations, actions, rewards, dones, next_observations = batch
        observations = self.preprocess_input_image(observations)
        next_observations = self.preprocess_input_image(next_observations)

        observations_tensor = torch.tensor(observations, dtype=torch.float, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.float, device=self.device)
        next_observations_tensor = torch.tensor(next_observations, dtype=torch.float, device=self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float, device=self.device).reshape(rewards.shape[0], 1)

        next_value = self.value_model(next_observations_tensor)
        target_value = reward_tensor + self.discount_factor * next_value
        predict_value = self.value_model(observations_tensor)
        advance = target_value - predict_value

        action_size = actions.shape[1]
        policy_tensor = self.policy_model(observations_tensor).view(actions.shape[0], -1)[:,:action_size]
        policy_tensor = policy_tensor * 0.99 + 0.005  # prevent log explosion

        #policy_loss = torch.sum(-(actions_tensor * torch.log(policy_tensor) + (1-actions_tensor)*torch.log(1-policy_tensor)))*advance.detach()
        policy_loss = torch.sum(-actions_tensor * torch.log(policy_tensor)) * advance.detach()
        policy_loss = policy_loss.mean()
        # print("Policy loss: ", policy_loss.detach().cpu().numpy())

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return {
            "loss": policy_loss.detach().cpu().numpy(),
        }

    def train_step(self, action, policy_tensor, value_tensor, next_observation, reward):
        actions_tensor = torch.zeros(22, dtype=torch.float, device=self.device)
        actions_tensor[:action.shape[0]] = torch.tensor(action, dtype=torch.float, device=self.device)

        next_observation = self.preprocess_input_image(next_observation)
        next_observation_tensor = torch.tensor(next_observation, dtype=torch.float, device=self.device).reshape(
            (-1, *next_observation.shape))
        reward_tensor = torch.tensor(reward, dtype=torch.float, device=self.device)

        next_value = self.value_model(next_observation_tensor)
        target_value = reward_tensor + self.discount_factor * next_value
        predict_value = value_tensor
        advance = target_value.detach() - predict_value

        value_loss = (target_value - predict_value).pow(2)
        # print("Value loss: ", value_loss.detach().cpu().numpy())

        policy_tensor = policy_tensor * 0.99 + 0.005  # prevent log explosion
        key_press_loss = torch.sum(actions_tensor) * self.key_press_penalty

        #policy_loss = torch.sum(-(actions_tensor * torch.log(policy_tensor) + (1 - actions_tensor) * torch.log(1 - policy_tensor))) * advance.detach()
        policy_loss = torch.sum(-actions_tensor*torch.log(policy_tensor)) * advance.detach()
        # print("Policy loss: ", policy_loss.detach().cpu().numpy())

        loss = value_loss + policy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env, observation_callback=None):

        observation = env.reset()
        observation = self.preprocess_input_image(observation)

        action_size = env.action_space.n
        input_size = self.policy_model.input_size

        for step in itertools.count():
            #env.render()

            observation_tensor = torch.tensor(observation, dtype=torch.float, device=self.device).reshape((-1, *observation.shape))
            policy_tensor = self.policy_model(observation_tensor).view(-1)

            if np.random.uniform() < self.random_chance:
                action = np.asarray(env.action_space.sample())
            else:
                action = policy_tensor.round().detach().cpu().numpy()[:action_size]

            actions_tensor = torch.zeros(22, dtype=torch.float, device=self.device)
            actions_tensor[:action_size] = torch.tensor(action, dtype=torch.float, device=self.device)

            #print(action)
            next_observation, reward, done, info = env.step(action)

            next_observation = self.preprocess_input_image(next_observation)
            next_observation_tensor = torch.tensor(next_observation, dtype=torch.float, device=self.device).reshape((-1, *next_observation.shape))
            reward_tensor = torch.tensor(reward, dtype=torch.float, device=self.device)

            next_value = self.value_model(next_observation_tensor)
            target_value = reward_tensor + self.discount_factor * next_value
            predict_value = self.value_model(observation_tensor)
            advance = target_value.detach() - predict_value

            value_loss = (target_value - predict_value).pow(2)
            #print("Value loss: ", value_loss.detach().cpu().numpy())

            policy_tensor = policy_tensor * 0.99 + 0.005    # prevent log explosion
            key_press_loss = torch.sum(actions_tensor) * self.key_press_penalty

            #policy_loss = torch.sum(-(actions_tensor*torch.log(policy_tensor)+(1-actions_tensor)*torch.log(1-policy_tensor))) * advance.detach()
            policy_loss = torch.sum(-actions_tensor*torch.log(policy_tensor)) * advance.detach()
            #print("Policy loss: ", policy_loss.detach().cpu().numpy())

            loss = value_loss + policy_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % 2000 == 0:
                #env.render()

                self.normalize_weight()

                print("Value loss: ", value_loss.detach().cpu().numpy())
                print("Policy loss: ", policy_loss.detach().cpu().numpy())
                self.save_model()

            if done:
                observation = env.reset()
                observation = self.preprocess_input_image(observation)
            else:
                observation = next_observation

    def preprocess_input_image(self, observation):
        input_img = np.asarray(observation).swapaxes(-1, -3) / 255.0
        return input_img

    def translate_actions_f(self, actions):
        ds = []
        for direction in actions[:, 4:8]:
            s = ""
            for i in direction:
                if i > 0.9:
                    s += "1"
                else:
                    s += "0"
            ds.append(s)
        direction_map = {
            "1000": 1,
            "1100": 2,
            "0100": 3,
            "0110": 4,
            "0010": 5,
            "0011": 6,
            "0001": 7,
            "1001": 8,
        }
        directions = np.asarray([direction_map.get(s, 0) for s in ds])

        actions_idx = directions + actions[:, 8] * 9 + actions[:, 9] * 18

        return actions_idx

    def translate_actions_b(self, actions_onehot):
        actions_idx = [np.where(a >= 0.9) for a in actions_onehot]
        direction_map = {
            0: [0, 0, 0, 0],
            1: [1, 0, 0, 0],
            2: [1, 1, 0, 0],
            3: [0, 1, 0, 0],
            4: [0, 1, 1, 0],
            5: [0, 0, 1, 0],
            6: [0, 0, 1, 1],
            7: [0, 0, 0, 1],
            8: [1, 0, 0, 1],
        }
        actions_res = [direction_map[idx%9].append(int(idx%9==0)).append(int(idx%18==0)) for idx in actions_idx]
        return actions_res

    def normalize_weight(self):
        with torch.no_grad():
            # self.common_cnn.conv3.weight.div_(torch.norm(self.common_cnn.conv3.weight, dim=1, keepdim=True))
            self.value_model.fc1.weight.div_(torch.norm(self.value_model.fc1.weight, dim=1, keepdim=True))
            self.value_model.fc2.weight.div_(torch.norm(self.value_model.fc2.weight, dim=1, keepdim=True))
            self.value_model.fc3.weight.div_(torch.norm(self.value_model.fc3.weight, dim=1, keepdim=True))
            self.policy_model.fc1.weight.div_(torch.norm(self.policy_model.fc1.weight, dim=1, keepdim=True))
            self.policy_model.fc2.weight.div_(torch.norm(self.policy_model.fc2.weight, dim=1, keepdim=True))
            self.policy_model.fc3.weight.div_(torch.norm(self.policy_model.fc3.weight, dim=1, keepdim=True))

    def save_model(self):
        torch.save(self.common_cnn.state_dict(), self.save_path + "common_cnn")
        torch.save(self.policy_model.state_dict(), self.save_path + "policy_model")
        torch.save(self.value_model.state_dict(), self.save_path + "value_model")

        torch.save(self.optimizer.state_dict(), self.save_path + "optimizer")

        print("Model saved.")

    def load_model(self):
        common_cnn_savefile = self.save_path + "common_cnn"
        policy_model_savefile = self.save_path + "policy_model"
        value_model_savefile = self.save_path + "value_model"
        optimizer_savefile = self.save_path + "optimizer"

        if os.path.isfile(common_cnn_savefile):
            self.common_cnn.load_state_dict(torch.load(common_cnn_savefile))
            print("CNN loaded.")
        if os.path.isfile(policy_model_savefile):
            self.policy_model.load_state_dict(torch.load(policy_model_savefile))
            print("Policy model loaded.")
        if os.path.isfile(value_model_savefile):
            self.value_model.load_state_dict(torch.load(value_model_savefile))
            print("Value model loaded.")

        if os.path.isfile(optimizer_savefile):
            self.optimizer.load_state_dict(torch.load(optimizer_savefile))
            print("Optimizer loaded.")
