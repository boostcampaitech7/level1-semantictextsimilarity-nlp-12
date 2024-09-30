import urllib.request
import json
import pandas as pd

def back_translate_papago(text, _from, _to):
  """
  역번역(파파고)
  """
  url = 'https://naveropenapi.apigw.ntruss.com/nmt/v1/translation' 
  client_id = '@#$' 
  client_secret = '@#$'

  data =\
  "source=" + _from + \
  "&target=" + _to + \
  "&text=" + urllib.parse.quote(text) + \
  "&honorific=true"

  query = 'query=' + urllib.parse.quote(text)

  request = urllib.request.Request(url)
  request.add_header("X-NCP-APIGW-API-KEY-ID", client_id)
  request.add_header("X-NCP-APIGW-API-KEY", client_secret)

  response = urllib.request.urlopen(request, data=data.encode("utf-8")) 
  rescode = response.getcode() 

  if rescode==200: 
    response_body = json.loads(response.read())
    return response_body
  else: 
    raise Exception("Error Code:", rescode)
  
def using_back_translate(text):
  """
  한국어 - 일본어 - 한국어 번역
  """
  res = back_translate_papago(text, "ko", "ja")
  translated_jp = res["message"]["result"]["translatedText"]
  res_2 = back_translate_papago(translated_jp, "ja", "ko") 
  return res_2["message"]["result"]["translatedText"]

df = pd.read_csv('trained_augument_3.csv') 
extreme_values = df[df['label'].isin([0.5, 1.5, 2.5, 4.5, 4.8])] 

extreme_values['augmented_sentence_1'] = extreme_values['sentence_1'].apply(lambda x: using_back_translate(x))
extreme_values['augmented_sentence_2'] = extreme_values['sentence_2'].apply(lambda x: using_back_translate(x))

augmented_data = pd.concat([extreme_values, extreme_values[['id', 'source', 'augmented_sentence_1', 'augmented_sentence_2', 'label', 'binary-label']].rename(columns={'augmented_sentence_1': 'sentence_1', 'augmented_sentence_2': 'sentence_2'})])

df_combined = pd.concat([df, augmented_data]) 
df_combined = df_combined.drop(columns=['augmented_sentence_1', 'augmented_sentence_2'], errors='ignore')

output_file_path = "trained_augument_back_translation.csv" 
df_combined.to_csv(output_file_path, index=False) 