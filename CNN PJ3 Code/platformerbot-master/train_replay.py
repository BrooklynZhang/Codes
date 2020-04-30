#PlatformBotProj
from PlatformerBot.bot import PlatformBot as Bot
from PlatformerBot.trainer import Trainer
from PlatformerBot.replay import Replay
import os

if __name__ == "__main__":
    bot = Bot()
    trainer = Trainer(bot=bot)

    files = []
    replay_folder = "replays/"
    for filename in os.listdir(replay_folder):
        files.append(os.path.join(replay_folder, filename))

    replays = []
    for file in files:
        with open(file, "rb") as replay_file:
            replay = Replay.load(replay_file)
            replays.append(replay)

    trainer.train_with_replays(replays)
