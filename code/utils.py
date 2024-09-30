import re

def only_completion(self, dataframe):
    """
    완성형 한글 음절 추출
    """
    def clean_text(text):
        pattern = r"[^가-힣\s]"
        cleaned_text = re.sub(pattern, '', text)
        return cleaned_text
    for text_column in self.text_columns:
        dataframe[text_column] = dataframe[text_column].apply(clean_text)
        
    return dataframe

def reduce_char(text):
    '''
    특수문자 줄이기
    참고( https://f7project.tistory.com/383 )
    '''
    
    text = re.sub('(\\S)\\1+', '\\1', text)    #반복되는 문자 (ㅋㅋㅋ->ㅋ, !!!->!)
    text = re.sub('([가-힣])\\1+', '\\1', text)   #반복되는 단어 (하하 -> 하)
    text = re.sub(r'\b(\w{1,})\1\b', r'\1', text)    #반복되는 n자리 이상 단어 (뿌듯뿌듯 -> 뿌듯)
    
    return text