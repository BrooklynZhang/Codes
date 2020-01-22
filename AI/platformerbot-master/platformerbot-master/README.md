# Platformer Bot

Actor-Critic based machine learning agent that can play retro games :D

## How to run

Make sure you have python 3 and pip installed. Install the required package using this command:

```shell
pip3 install -r requirements.txt
```

To run the GUI, execute `run.py` directly.

```shell
python3 run.py
```

To train the network (without GUI), run `train.py`.

To train the network with replays (without GUI), run `train_replay.py`.

## About the GUI

The GUI is mostly self explanatory. Click the button at the top left to change mode. While in interactive mode, you can control the game using WSAD and J,K.

Click record at any time to start recording. The button will be highlighted while recording. Click it again to stop recording. Switch to playback mode to watch the recorded replay. Select the replay and click load can load the selected replay. 

You can switch to train mode to watch the agent train in real time. It is not very efficient because every frame is rendered to the screen, but it is pretty fun to watch. Switch to demo mode to watch the agent play without train.

Click visualize network button anytime will feed the current frame into the model and visualize it. Switch to forward tab at the right side panel to see the visualization result.