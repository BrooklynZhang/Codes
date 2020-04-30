#PlatformBotProj
from PlatformerBot.bot import PlatformBot
import retro


if __name__ == "__main__":
    bot = PlatformBot()
    env = retro.make(game="SuperMarioBros-Nes")
    bot.train(env)