import pandas as pd
import urllib.request
import json
import pandas as pd
import random 
from konlpy.tag import Okt

random.seed(0)

def undersampling():
    """
    Under Sampling 통해 0점 데이터 개수 줄이기 (750개 제거)
    """
    df = pd.read_csv('train.csv')
    df_0 = df[df['label']==0][0:1500].copy()

    df_new = df[df['label']!=0].copy()
    df_new = pd.concat([df_new, df_0])
    df_new


def swap_sentence():
    """
    swap sentence
    """
    df_switched = pd.read_csv('train.csv')
    df_switched["sentence_1"] = df["sentence_2"]
    df_switched["sentence_2"] = df["sentence_1"]
    df_switched = df_switched[df_switched['label'] != 0]
    df_switched


def copied_sentence():
    """
    copied sentence
    """
    copied_df = df[df['label']==0][1500:].copy()
    copied_df['sentence_1'] = copied_df['sentence_2']
    copied_df['label'] = 5.0
    copied_df

def synonym_replacement(path_src='.', path_dst='.', ratio='0.2'):
    synonym_dict = {
        '너무': ['매우', '굉장히', '아주', '엄청', '완전'],
        'person': ['사람', '인물'],
        '정말': ['진짜', '참으로', '매우', '참', '진심으로'],
        'ㅋㅋ': ['ㅎㅎ', '하하', 'ㅋㅋㅋ', 'ㅎㅎㅎ', '재미 있다'],
        'ㅎㅎ': ['ㅋㅋ', '하하', 'ㅋㅋㅋ', 'ㅎㅎㅎ', '재미 있다'], 
        'ㅋㅋㅋ': ['ㅎㅎ', '하하', 'ㅋㅋ', 'ㅎㅎㅎ', '재미 있다'],
        'ㅎㅎㅎ': ['ㅎㅎ', '하하', 'ㅋㅋ', 'ㅋㅋㅋ', '재미 있다'],
        '저도': ['나도', '저 역시', '마찬가지로'],
        '함께': ['같이', '동반하여', '더불어'],
        '우리': ['저희', '함께', '공동으로'],
        '있습니다': ['존재합니다', '보유하고 있습니다', '있어요'],
        '다시': ['재차', '또', '다시 한 번'],
        '좋은': ['훌륭한', '멋진', '최고의', '괜찮은'],
        '감사합니다': ['고맙습니다', '감사의 말씀을 드립니다', '감사해요'],
        '많은': ['다수의', '풍부한', '대량의', '다양한'],
        '이야기': ['대화', '담소', '토크', '얘기'],
        '다음': ['이후', '다음번', '차후'],
        
        # 영화 관련
        '영화': ['필름', '작품', '무비', '시네마'],
        '드라마': ['시리즈', 'TV 프로그램', '연속극'],
        '재미있게': ['흥미롭게', '즐겁게', '재밌게'],
        '재밌다': ['재미있다', '흥미롭다', '즐겁다'],
        '눈물이': ['감동이', '감격이', '슬픔이'],
        '평점': ['별점', '리뷰 점수', '평가 점수'],
        '보고': ['감상하고', '시청하고'],
        '봤는데': ['관람했는데', '시청했는데'],
        
        # 청원 및 정치 관련
        '청원합니다': ['요청합니다', '탄원합니다', '호소합니다'],
        '주세요': ['해주세요', '부탁드립니다', '요청합니다'],
        '합니다': ['실행합니다', '수행합니다', '이행합니다'],
        '폐지': ['철폐', '없애기', '종료'],
        '청원': ['요청', '탄원', '청구'],
        '국회의원': ['의원', '정치인', '입법자'],
        '반대': ['거부', '불찬성', '반박'],
        '반대합니다': ['거부합니다', '반대 의견을 표명합니다'],
        '관련': ['연관된', '관계된', '해당하는'],
        '제발': ['부디', '꼭', '간곡히'],
        '부동산': ['real estate', '토지 및 건물', '재산'],
        '요청합니다': ['신청합니다', '청구합니다', '요구합니다'],
        '청와대': ['대통령 집무실', '대통령 관저'],
        '대통령': ['국가원수', '최고지도자', '대표자'],
        '해주세요': ['실행해 주십시오', '조치해 주십시오'],
        '출국금지': ['출국제한', '출국 불허'],
        '바랍니다': ['희망합니다', '원합니다', '요망합니다'],
        
        # 기타
        'person': ['개인', '사람', '인물'],
        '회사': ['기업', '법인', '직장'],
        '오늘': ['금일', '당일', '현재'],
        '없다': ['부재하다', '존재하지 않는다', '결여되다'],
        '그냥': ['단순히', '그저', '별다른 이유 없이'],
        '오랜만에': ['久しぶりに', '오래간만에', '한참 만에'],
        '이렇게': ['이와 같이', '이런 식으로', '이러한 방식으로'],
        '완전': ['전적으로', '완벽하게', '철저히'],
        '내가': ['제가', '나 자신이', '본인이'],
        '대한': ['관한', '대해', '관련된'],
        '위한': ['목적으로', '위해서', '~을 위하여'],
        '가상화폐': ['암호화폐', '디지털 화폐', '가상 통화']
    }
    
    def get_synonyms(word):
        lemmas = None
        if word in synonym_dict.keys():
            lemmas = random.choice(synonym_dict[word])
        return lemmas
    
    def synonym_replacement(sentence, replacement_prob=0.2):
        words = okt.morphs(sentence)  # 문장을 단어로 분리
        new_words = []
        
        for word in words:
            if random.uniform(0, 1) < replacement_prob:  # 20% 확률로 교체 시도
                synonyms = get_synonyms(word)
                if synonyms:
                    new_word = random.choice(synonyms)  # 동의어 중 하나를 무작위로 선택
                    new_words.append(new_word)
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    okt = Okt()
    df = pd.read_csv(path_src)
    num_rows_to_augment = int(ratio * len(df))
    indices_to_augment = random.sample(range(len(df)), num_rows_to_augment)

    augmented_data = []
    
    for idx in indices_to_augment:
        row = df.iloc[idx]
        sentence_1_aug = synonym_replacement(row['sentence_1'])
        sentence_2_aug = synonym_replacement(row['sentence_2'])
        augmented_data.append([row['id'], row['source'], sentence_1_aug, sentence_2_aug, row['label'], row['binary-label']])

    augmented_df = pd.DataFrame(augmented_data, columns=df.columns)
    final_df = pd.concat([df, augmented_df])
    final_df.to_csv(path_dst, index=False)