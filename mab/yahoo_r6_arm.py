import sqlite3
import pandas as pd
import numpy as np


class ArticleArms(object):
    """
    Arms of Yahoo! R6 dataset
    """
    def __init__(self, path_db, num_cache_lines=10000):
        self.conn = sqlite3.connect(path_db)
        self.cursor = self.conn.cursor()
        self.event_cache = None
        self.num_cache_lines = num_cache_lines
        self.event_id = 1

        num_total_events_df = pd.read_sql_query('SELECT count(*) FROM event', self.conn)
        self.num_total_events = num_total_events_df.iloc[0].values[0]
        self.article_df =\
            pd.read_sql_query('SELECT articleID, feat1, feat2, feat3, feat4, feat5, feat6 FROM article', self.conn)
        self.pool_article_df = pd.read_sql_query('SELECT * FROM poolarticle', self.conn)
        self.counts_for_pools = self.pool_article_df.groupby('poolID').size()
        rejected_articles_df =\
            pd.read_sql_query('SELECT articleID FROM article where rejected=1', self.conn)  # pandas DataFrame
        self.rejected_articles = rejected_articles_df['articleID'].values  # numpy ndarray

        self.current_event_df = None
        self.current_article = None
        self.current_user_features = None
        self.current_user_cluster = None
        self.previous_pool = None
        self.current_pool = None
        self.current_pool_size = None
        self.current_pool_articles = None
        self.current_pool_articles_features_df = None
        self.current_article_rejected = None
        self.current_article_features = None

        self.current_num_arms = None
        self.previous_arm_names = np.array([])
        self.current_arm_names = None
        self.current_arm_index = None
        self.current_reward = None
        self.arm_changed = False

        self.next()

    def __del__(self):
        self.cursor.close()
        self.conn.close()

    """
    function next(self): move to the next state (or next event)
    returns: (num_arms_added, removing_arms_indices)
    - num_arms_added: # of arms should be added in the next time
    - removing_arms_indices: indices of arms, should be removed from the current arms
    """
    def next(self):
        num_arms_added = 0
        removing_arms_indices = []

        self.current_event_df = self.__get_event()  # pandas Series
        self.current_article = int(self.current_event_df['articleID'])  # int

        rejected = np.in1d(self.current_article, self.rejected_articles)[0]  # numpy ndarray
        if rejected:
            return self.next()

        self.current_pool = int(self.current_event_df['poolID'])  # int

        if self.previous_pool != self.current_pool:
            self.current_pool_size = self.counts_for_pools[self.current_pool]  # int
            self.current_pool_articles = \
                self.pool_article_df[self.pool_article_df['poolID'] ==
                                     self.current_pool]['articleID'].values  # numpy ndarray
            self.current_pool_articles_features_df =\
                self.article_df[self.article_df['articleID'].isin(self.current_pool_articles)]  # pandas DataFrame

            if len(self.previous_arm_names) == 0:  # if starts
                self.current_arm_names = self.current_pool_articles
            else:
                adding_arms = np.setdiff1d(self.current_pool_articles, self.previous_arm_names)
                num_arms_added = adding_arms.size
                removing_arms = np.setdiff1d(self.previous_arm_names, self.current_pool_articles)
                removing_arms_indices =\
                    np.arange(self.previous_arm_names.size)[np.in1d(self.previous_arm_names, removing_arms)]
                removing_arms_indices = removing_arms_indices.tolist()  # python list
                self.current_arm_names = np.append(self.current_arm_names, adding_arms)
                self.current_arm_names = np.delete(self.current_arm_names, removing_arms_indices)

            self.previous_arm_names = self.current_arm_names
            self.current_num_arms = len(self.current_arm_names)

        self.previous_pool = self.current_pool
        target_article =\
            self.current_pool_articles_features_df[self.current_pool_articles_features_df['articleID'] ==
                                                   self.current_article].iloc[0].values
        self.current_user_features = self.current_event_df['userFeat1':'userFeat6'].values  # numpy ndarray
        self.current_user_cluster = int(self.current_event_df['userCluster'])  # int
        self.current_article_features = target_article[1:7]  # numpy ndarray (index 1~6)
        self.current_arm_index, = np.where(self.current_arm_names == self.current_article)[0]
        self.current_reward = int(self.current_event_df['click'])

        if self.event_id > 1 and (num_arms_added != 0 or bool(removing_arms_indices)):
            self.arm_changed = True
        else:
            self.arm_changed = False

        return num_arms_added, removing_arms_indices

    def get_num_arms(self):
        return self.current_num_arms

    def get_arm_index(self):
        return self.current_arm_index

    def get_user_features(self):
        return self.current_user_features

    def get_article_features(self, arm):
        return self.current_pool_articles_features_df[self.current_pool_articles_features_df['articleID'] ==
                                                      self.current_arm_names[arm]].iloc[0].values[1:7]

    def get_article_features_all(self):
        adjusted_indices =\
            np.searchsorted(self.current_pool_articles_features_df['articleID'].values, self.current_arm_names)
        return self.current_pool_articles_features_df.iloc[adjusted_indices].loc[:, 'feat1':'feat6'].as_matrix()

    def get_hybrid_features(self, arm):
        return np.outer(self.current_user_features, self.get_article_features(arm)).flatten()

    def get_hybrid_features_all(self):
        all_hybrid_features = self.get_hybrid_features(0)
        for i in range(1, self.current_num_arms):
            all_hybrid_features = np.vstack((all_hybrid_features, self.get_hybrid_features(i)))
        return all_hybrid_features

    def get_all_features(self):
        return np.hstack((self.get_hybrid_features_all(), self.get_article_features_all()))

    def get_reward(self, arm):
        if self.current_arm_names[arm] == self.current_article:
            return int(self.current_event_df['click'])
        else:
            return 0

    def __get_event(self):
        if self.event_cache is None:
            query = """SELECT displayed as articleID, click, poolID, cluster as userCluster,
                       feat1 as userFeat1, feat2 as userFeat2, feat3 as userFeat3,
                       feat4 as userFeat4, feat5 as userFeat5, feat6 as userFeat6
                       FROM event LEFT JOIN user ON event.userID = user.userID
                       WHERE event.eventID >= ? AND event.eventID <= ?"""
            event_from = self.event_id
            event_to = event_from + self.num_cache_lines
            self.event_cache = pd.read_sql_query(query, self.conn, params=(event_from, event_to))

        remainder = self.event_id % self.num_cache_lines
        result_df = self.event_cache.iloc[remainder - 1]  # return type: pandas Series

        if remainder == 0:
            self.event_cache = None

        if self.event_id == self.num_total_events:
            self.event_id = 0
            self.event_cache = None

        self.event_id += 1
        return result_df
