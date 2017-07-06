from slacker import Slacker
import numpy as np
import random
import common_params
import datetime
import database


class SlackNewsBot:

    def __init__(self, common_params, is_update_db=False):
        self.param = common_params
        self.slack = Slacker(self.param.slack_token)
        self.df = common_params.load_news_df()
        self.attachments = []
        self.is_update_db = is_update_db
        self.db = database.NewsDatabase(self.param)

    def get_slack_message_for_selected_news(self, df_sub, is_featured):
        title = df_sub['title']
        description = df_sub['description']
        #original_url = self.news_main_url + df_sub[4]
        redirection_url = df_sub['redirection_url']
        thumbnail_url = df_sub['thumbnail_url']
        dic = dict()

        if is_featured:
            dic['image_url'] = thumbnail_url
            #dic['pretext'] = ''' *_[Featured News by A-Lab]_* '''
            dic['color'] = self.param.slack_color_dict[0]
            dic['author_name'] = 'Featured News'
            dic['author_link'] = 'http://www.hyundaicard.com'
            dic['author_icon'] = 'https://avatars3.githubusercontent.com/u/162998?v=3&s=88'
        else:
            dic['thumb_url'] = thumbnail_url

        #dic['color'] = self.news_subject_dict[subject_idx][1]

        dic['title'] = title
        dic['title_link'] = redirection_url
        dic['fallback'] = description[0:self.param.description_length] + '...'
        dic['text'] = description[0:self.param.description_length] + '...'
        dic['footer'] = 'A-Lab News Bot'
        dic['mrkdwn_in'] = ["author_name", "text", "pretext"]

        return dic

    def append_attachments_per_subject(self, subject_idx, is_featured=False):
        df_selected_subject = self.df.loc[self.df['subject_idx'] == subject_idx]
        num_news_per_subject = len(df_selected_subject)
        selected_news_idx = np.random.choice(num_news_per_subject)
        df_selected_news = df_selected_subject.iloc[selected_news_idx]
        dic = self.get_slack_message_for_selected_news(df_selected_news, is_featured)
        self.attachments.append(dic)
        if self.is_update_db:
            self.db.insert_event(df_selected_news)

    def append_attachments_per_title(self, news_idx, is_featured=False):
        df_selected_news = self.df.iloc[news_idx]
        dic = self.get_slack_message_for_selected_news(df_selected_news, is_featured)
        self.attachments.append(dic)
        if self.is_update_db:
            self.db.insert_event(df_selected_news)

        return df_selected_news['subject_idx']

    def post_message(self, selected_arm, is_post_per_subject):
        featured_news_idx = selected_arm  # MAB output (selected arm index)

        # Option 1: 뉴스 주제별로 한 건씩 보여줌
        #   > MAB에서 선정한 1건을 맨 위에 보여줌
        #   > MAB로 기 선택된 뉴스가 다시 선택되지 않게 뉴스 후보 리스트군에서 제거하고
        #   > 나머지 8건의 뉴스 종류에서 num_news_display-1 건수만큼 랜덤으로 선택해 보여줌
        if is_post_per_subject:
            subject_group = (self.df.groupby('subject_idx')).size()
            subject_list = list(subject_group.index.values)
            featured_subject_idx = self.append_attachments_per_title(featured_news_idx, True)
            subject_list.remove(featured_subject_idx)
            subject_show_list = random.sample(subject_list, self.param.num_news_display-1)
            for v in subject_show_list:
                self.append_attachments_per_subject(v)
        # Option 2: 뉴스 주제와 무관하게 랜덤하게 보여줌
        #   > MAB에서 선정한 1건을 맨 위에 보여줌
        #   > MAB로 기 선택된 뉴스가 다시 선택되지 않게 뉴스 후보 리스트군에서 제거하고
        #   > 나머지 뉴스에서 num_news_display-1 건수만큼 랜덤으로 선택해 보여줌
        else:
            news_list = list(range(self.param.num_news))
            self.append_attachments_per_title(featured_news_idx, True)

            #self.append_attachments_df.iloc[2:]

            news_list.remove(featured_news_idx)
            news_show_list = random.sample(news_list, self.param.num_news_display-1)
            for v in news_show_list:
                self.append_attachments_per_title(v)

        return self.attachments


if __name__ == "__main__":
    is_post_per_subject = True
    is_update_db = False
    selected_arm = 0

    param = common_params.CommonParams()
    bot = SlackNewsBot(param, is_update_db)
    df = bot.df

    if is_update_db:
        bot.db.open_db()
        msg_attachments = bot.post_message(selected_arm, is_post_per_subject)
        bot.db.close_db()
    else:
        msg_attachments = bot.post_message(selected_arm, is_post_per_subject)

    msg_text = (datetime.datetime.now()).strftime('*_A-lab의 뉴스봇, %Y/%m/%d %H:%M_*')
    bot.slack.chat.post_message('@byonghwa.oh', msg_text,
                                attachments=msg_attachments, as_user=True)
