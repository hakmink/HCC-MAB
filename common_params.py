import pandas as pd

class CommonParams(object):
    """
    Common parameters for scrapping & displaying news
    """
    def __init__(self):
        # 리다이렉션 서버 주소 및 포트 번호
        self.server_address = "localhost"
        self.server_port = 8001

        # 네이버 모바일 뉴스 URL
        self.news_main_url = 'http://m.news.naver.com'

        # 뉴스 주제별 랭킹 뉴스 (총 주제 9건, 주제별 추천 뉴스 3건이므로 총 추천 뉴스 건수는 27건)
        self.news_subject = ['politics', 'economy', 'society', 'it_secience',
                        'life_culture', 'world', 'entertainments',
                        'photo_section', 'tv_section']

        # 뉴스 주제
        self.news_subject_dict = {0: '정치', 1: '경제', 2: '사회',
                                  3: 'IT', 4: '생활', 5: '세계',
                                  6: '연예', 7: '포토', 8: 'TV'}
        # 뉴스 주제별 color 정보
        self.slack_color_dict = {0: '#8904B1', 1: '#0404B4', 2: '#B92323',
                                 6: '#26FF92', 7: '#FF00BF', 8: '#424242'}

        # 뉴스 건수
        self.num_news_subject = len(self.news_subject_dict)
        self.num_news = 0

        # 사용자에게 보여줄 뉴스 건수
        self.num_news_display = 5

        # 뉴스 요약 길이
        self.description_length = 40

        # 로그 디렉토리
        self.log_dir = "log"

        # 뉴스 데이터를 저장할 csv 파일명
        self.csv_path = self.log_dir + '/news.csv'
        self.csv_path_old = self.log_dir + '/news_old.csv'

        self.pool_idx_path = self.log_dir + '/pool_idx.log'

        # 데이터베이스 파일명
        self.db_name = 'alab_news.db'

        # 네이버 주제별 랭킹 뉴스를 검색하기 위한 정규식 표현
        # 예: 정치 뉴스 => politics1, politics2, politics3
        self.news_subject_regexp = ''
        for k in range(0, len(self.news_subject)):
            self.news_subject_regexp += (self.news_subject[k] + '[0-9]|')
        self.news_subject_regexp = self.news_subject_regexp[:-1]

        # slack token
        self.slack_token = 'xoxb-147325464930-6nErkT2qD7iYPDoPlFJL0MuW'

        # Data-frame column name
        self.df_col_names = ['subject_idx', 'article_idx', 'title', 'description',
                             'news_url', 'redirection_url', 'oid', 'aid',
                             'title_length', 'visit', 'news_number_of_words', 'is_emphasis',
                             'num_omitted', 'news_creator', 'thumbnail_url']

    def load_news_df(self, is_update_num_news=True):
        df = pd.read_csv(self.csv_path, names=self.df_col_names)
        if is_update_num_news is True:
            subject_group = (df.groupby('subject_idx')).size()
            self.num_news = len(df)
            self.num_news_subject = len(subject_group)
        return df
