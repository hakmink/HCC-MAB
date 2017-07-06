import slack_news_bot
import common_params
import datetime
import util_funcs
from mab import algorithm as bd
from mab import arm
from mab import scorer as sc
import time
import pickle


is_post_per_subject = False
is_update_db = True

param = common_params.CommonParams()
bot = slack_news_bot.SlackNewsBot(param, is_update_db)
df = bot.df
num_arms = param.num_news
path = 'log/mab.log'

if util_funcs.check_file(path) is False:
    algorithm = bd.EpsilonGreedyAlgorithm(num_arms, 0.1)
    trials = 0
    with open(path, 'wb') as f:
        pickle.dump(algorithm, f)
        pickle.dump(trials, f)
    print("MAB started")
else:
    with open(path, 'rb') as f:
        algorithm = pickle.load(f)
        trials = pickle.load(f)
    print("MAB object loaded")
    print(trials)

selected_arm = algorithm.select_arm()

if is_update_db:
    bot.db.open_db()
    msg_attachments = bot.post_message(selected_arm, is_post_per_subject)
    bot.db.close_db()
else:
    msg_attachments = bot.post_message(selected_arm, is_post_per_subject)

msg_text = (datetime.datetime.now()).strftime('*_A-lab의 뉴스봇, %Y/%m/%d %H:%M_*')
bot.slack.chat.post_message('@byonghwa.oh', msg_text,
                            attachments=msg_attachments, as_user=True)
# bot.slack.chat.post_message('#lifeml_mab', msg_text,
#                            attachments=msg_attachments, as_user=True)
