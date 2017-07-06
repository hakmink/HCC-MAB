import sqlite3 as lite
import sys
from os.path import isfile, getsize
import pandas as pd
import util_funcs
import common_params
import datetime


class NewsDatabase(object):

    def __init__(self, common_params):
        self.param = common_params
        self.conn = None

    def create_db(self):
        sql_create_table_article = """ CREATE TABLE IF NOT EXISTS article (
                                            articleID INTEGER NOT NULL,
                                            subjectID INTEGER NOT NULL,
                                            rejected BOOLEAN,
                                            feat1 FLOAT,
                                            feat2 FLOAT, 
                                            feat3 FLOAT,
                                            feat4 FLOAT,
                                            feat5 FLOAT, 
                                            feat6 FLOAT,
                                            PRIMARY KEY(articleID)
                                        ); """

        sql_create_table_event = """ CREATE TABLE IF NOT EXISTS event (
                                            eventID INTEGER PRIMARY KEY AUTOINCREMENT,
                                            datetime DATETIME,
                                            displayed INTEGER, 
                                            click INTEGER,
                                            poolID INTEGER NOT NULL,
                                            userID INTEGER,
                                            FOREIGN KEY(displayed) REFERENCES article (articleID),
                                            FOREIGN KEY(poolID) REFERENCES pool (poolID)
                                            FOREIGN KEY(userID) REFERENCES user (userID) 
                                        ); """

        sql_create_table_pool = """CREATE TABLE IF NOT EXISTS pool (
                                        poolID INTEGER PRIMARY KEY
                                    );"""

        sql_create_table_poolarticle = """CREATE TABLE IF NOT EXISTS poolarticle (
                                            poolID INTEGER NOT NULL,
                                            articleID INTEGER NOT NULL,
                                            PRIMARY KEY(poolID, articleID),
                                            FOREIGN KEY(poolID) REFERENCES pool(poolID),
                                            FOREIGN KEY(articleID) REFERENCES article(articleID)
                                    );"""

        sql_create_table_user = """CREATE TABLE IF NOT EXISTS user (
                                            userID INTEGER PRIMARY KEY AUTOINCREMENT,
                                            cluster INTEGER,
                                            feat1 FLOAT, 
                                            feat2 FLOAT, 
                                            feat3 FLOAT, 
                                            feat4 FLOAT,
                                            feat5 FLOAT 
                                    );"""

        self.conn = lite.connect(self.param.db_name)
        cur = self.conn.cursor()
        cur.execute(sql_create_table_article)
        cur.execute(sql_create_table_event)
        cur.execute(sql_create_table_pool)
        cur.execute(sql_create_table_poolarticle)
        cur.execute(sql_create_table_user)
        print("Tables created successfully.")
        self.close_db()

    def open_db(self):
        if util_funcs.check_db_file(self.param.db_name) == 0:
            print("Creating New Database..")
            self.create_db()

        self.conn = lite.connect(self.param.db_name)


    def close_db(self):
        self.conn.commit()
        self.conn.close()

    def insert_article(self, pool_idx):
        df = self.param.load_news_df()

        # DB에서 사용할 article ID 포맷: [oid][aid]
        df_article_id = df.iloc[:, 6].astype(str) + df.iloc[:, 7].astype(str)
        #df_article_id = df.iloc[:, 7].astype(str)
        # subject ID column
        df_article = pd.concat([df_article_id, df[['subject_idx']].astype(str)], axis=1)

        cur = self.conn.cursor()

        # 뉴스 기사 삽입
        cur.executemany("INSERT OR REPLACE INTO article VALUES(?, ?, 0, 1, 1, 1, 1, 1, 1)",
                        list(df_article.to_records(index=False)))

        cur.execute('SELECT COUNT(*) FROM pool')
        db_pool_idx = int(cur.fetchone()[0])
        print('db_pool_idx={}, pool_idx={}'.format(db_pool_idx, pool_idx))

        # pool ID 증가 시, 신규 뉴스 pool 생성 및 삽입
        if db_pool_idx == 0 or db_pool_idx != pool_idx:
            cur.execute("INSERT INTO pool VALUES (?)", (pool_idx,))

            for k in range(self.param.num_news):
                cur.execute("INSERT OR REPLACE INTO poolarticle VALUES(?,?)", (pool_idx, df_article[0][k]))
            print("Pool changed")
            return True
        else:
            print("Pool unchanged")
            return False

    def get_recent_pool_idx(self):
        cur = self.conn.cursor()
        cur.execute('SELECT COUNT(*) FROM pool')
        pool_idx = cur.fetchone()[0]
        return pool_idx

    def insert_event(self, df_selected_news):
        # 이 메소드를 정확히 구현하기 위해서는 사용자가 접속하고 있을 때의 이벤트 핸들러도 받아야 할 듯 함..
        # (현재로서는 링크를 클릭했을 때의 이벤트 핸들러만 받아올 수 있으므로)
        pool_idx = self.get_recent_pool_idx()
        cur = self.conn.cursor()

        date_time = (datetime.datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
        article_id = df_selected_news['oid'].astype(str) + df_selected_news['aid'].astype(str)

        sql_insert = "INSERT INTO event VALUES (NULL, ?, ?, 0, ?, NULL)"
        cur.execute(sql_insert, (date_time, article_id, pool_idx))


if __name__ == "__main__":
    param = common_params.CommonParams()
    db = NewsDatabase(param)
    db.open_db()
    #db.insert_article(1)
    db.close_db()
