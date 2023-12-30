# What's your EDA?

# 터미널에 python EDA.py 
# 입력하면 됩니다

# 시간 좀 걸립니다
# 최초 1번만 돌리서야 됩니다!!!!!!!(범주화)


# 모듈 import
import pandas as pd
import numpy as np
from collections import Counter
import re

# Data load
data_path='data/'

users = pd.read_csv(data_path + 'users.csv')
books = pd.read_csv(data_path + 'books.csv')
ratings = pd.read_csv(data_path+'train_ratings.csv')


# eda 잘 되었는지 확인하는 함수
def eda_prograss(users,books):
    #1.location(결측치 확인하는거고 적을 수록 좋음)
    print("1:location 결측치:", users['location_country'].apply(lambda x: isinstance(x, float) and np.isnan(x)).sum())

    #2.age(결측치확인하는거고 0나와야됨)
    print("2:age 결측치:",users['age'].isna().sum())

    #3.year_of_publication(6개로 범주화하였기에 6 나와야됨)
    print("3:year_of_publication 범주 갯수:", books['year_of_publication'].nunique())

    #4.language(결측치 0나와야됨)
    print("4:language 결측치:", books['language'].isnull().sum())

    #5.language(주요 언어 4개와 나머지로 분류)
    print("5:language 범주 갯수:", books['language'].nunique())
    
    #6.category(카테고리 개수이고 많이 줄여야 좋음)
    print("6:category 범주 갯수:", books['category'].nunique())

    #7.category(결측치 적을수록 좋음)
    print("7:category 결측치:", books['category'].isnull().sum())

    #8.summary (summary가 없을 시 title로 대체이고 0나와야함)
    print("8:summary 결측치:", books['summary'].isnull().sum())

    #9. publisher (11571->1522)
    print("9:publisher 범주 갯수:", books['publisher'].nunique())

    print("--------------------------------------------")

#------------------------------------------------------------------------------------------
### users
## 1) location
# user data에서 location 구분
users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '', regex=True) # 특수문자 제거

users['location_city'] = users['location'].apply(lambda x: x.split(',')[0].strip())
users['location_state'] = users['location'].apply(lambda x: x.split(',')[1].strip())
users['location_country'] = users['location'].apply(lambda x: x.split(',')[2].strip())

users = users.replace('na', np.nan) #특수문자 제거로 n/a가 na로 바뀌게 되었습니다. 따라서 이를 컴퓨터가 인식할 수 있는 결측값으로 변환합니다.
users = users.replace('', np.nan) # 일부 경우 , , ,으로 입력된 경우가 있었으므로 이런 경우에도 결측값으로 변환합니다.

# eda 하기전 상태 확인
eda_prograss(users,books)

# country가 결측값인 일부 행을 살펴보면 city값이 존재하는데, country 정보가 없는 경우 처리
modify_location = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values
location = users[(users['location'].str.contains('seattle'))&(users['location_country'].notnull())]['location'].value_counts().index[0]

location_list = []
for location in modify_location:
    try:
        right_location = users[(users['location'].str.contains(location))&(users['location_country'].notnull())]['location'].value_counts().index[0]
        location_list.append(right_location)
    except:
        pass
for location in location_list:
    users.loc[users[users['location_city']==location.split(',')[0]].index,'location_state'] = location.split(',')[1]
    users.loc[users[users['location_city']==location.split(',')[0]].index,'location_country'] = location.split(',')[2]


#------------------------------------------------------------------------------------------
## 2) age
# location_country별 국가별 age 평균 계산
country_mean_age = users.groupby('location_country')['age'].mean()
global_mean_age = users['age'].mean()

# age가 결측치인 행 찾기
missing_age_rows = users['age'].isnull()

# 결측치를 해당 국가의 평균으로 채우기
users.loc[missing_age_rows, 'age'] = users.loc[missing_age_rows, 'location_country'].map(country_mean_age)

# 남아있는 결측치를 전체 평균으로 채우기
users['age'].fillna(global_mean_age, inplace=True)

# user의 age를 반올림
users["age"] = users["age"].round()

## age 범주화는 contect.py 파일에서 진행

#------------------------------------------------------------------------------------------
### books

## 3) year_of_publication
# 연도별로 범주화하여 대체하는 함수
def categorize_and_replace(year):
    if year <= 1990:
        return 1
    elif 1990 < year <= 1993:
        return 2
    elif 1993 < year <= 1996:
        return 3
    elif 1996< year <= 1999:
        return 4
    elif 1999< year <= 2002:
        return 5
    else:
      return 6

# 'year_of_publication' 열 대체
books['year_of_publication'] = books['year_of_publication'].apply(categorize_and_replace)

#------------------------------------------------------------------------------------------

## 4,5) language 결측치 처리 및 범주화
# 파악된 갯수 데이터를 바탕으로 갯수가 10개 미만인 것은 ot로 분류하기로 하였다.
lang_dict = {'0': 'en', '3': 'de', '8': 'es', '2': 'fr'}
lang_list = ['en', 'de', 'es', 'fr', 'it','nl','pt','da', 'ca',  'ms']
def fill_lang(row):
   if row['language'] in lang_list:
         return row['language']
   isbn_start = row['isbn'][:1]
   if pd.isnull(row['language']) and isbn_start in lang_dict:
           return lang_dict[isbn_start]
   return 'ot'

books['language'] = books.apply(fill_lang, axis=1)


#------------------------------------------------------------------------------------------
## 6) category 범주화
# category 전처리
books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
books['category'] = books['category'].str.lower()

books['category_high'] = books['category'].copy()

categories = ['garden','crafts','physics','adventure','music','fiction','nonfiction','science','science fiction','social','homicide',
 'sociology','disease','religion','christian','philosophy','psycholog','mathemat','agricult','environmental',
 'business','poetry','drama','literary','travel','motion picture','children','cook','literature','electronic',
 'humor','animal','bird','photograph','computer','house','ecology','family','architect','camp','criminal','language','india']

for category in categories:
    books.loc[books[books['category'].str.contains(category,na=False)].index,'category_high'] = category
category_high_df = pd.DataFrame(books['category_high'].value_counts()).reset_index()
category_high_df.columns = ['category','count']

# 100개 이하인 항목은 others로 묶어주도록 하겠습니다.
others_list = category_high_df[category_high_df['count']<100]['category'].values

books.loc[books[books['category_high'].isin(others_list)].index, 'category_high']='others'
books['category'] = books['category_high']


#------------------------------------------------------------------------------------------
## 7) category 결측치 채우기 
# 하나의 df생성
df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')

# category 컬럼에 결측치가 있는 isbn 식별
missing_category_isbns = books[pd.isnull(books['category'])]['isbn']

# 각 결측치가 있는 isbn에 대해 가장 많이 읽은 category 찾기
for isbn in missing_category_isbns:
    # 해당 isbn을 읽은 사용자들 찾기
    users_who_read_book = df[df['isbn'] == isbn]['user_id']

    # 이 사용자들이 읽은 다른 책들의 category 확인
    categories_read_by_these_users = books[books['isbn'].isin(df[df['user_id'].isin(users_who_read_book)]['isbn'])]['category']

    # 가장 많이 읽은 category 확인
    most_common_category = Counter(categories_read_by_these_users.dropna()).most_common(1)
    if most_common_category:
        most_common_category = most_common_category[0][0]

        # 해당 isbn의 결측치를 가장 많이 읽은 category로 채우기
        books.loc[books['isbn'] == isbn, 'category'] = books.loc[books['isbn'] == isbn, 'category'].fillna(most_common_category)

# 남은 결측치 unknown으로 대체
books['category'] = books['category'].fillna('unknown')


#------------------------------------------------------------------------------------------
## 8) summary 결측치 채우기
# 'summary' 열의 결측값 확인
missing_summary = books['summary'].isnull()

# 결측값인 'summary' 값을 해당하는 'book_title' 값으로 대체
books.loc[missing_summary, 'summary'] = books.loc[missing_summary, 'book_title']

#------------------------------------------------------------------------------------------
## 9) publisher 범주화
# 출판사 등장 횟수를 담은 사전을 데이터프레임으로 변환하고, 열 이름은 'publisher'와 'count
publisher_dict=(books['publisher'].value_counts()).to_dict()
publisher_count_df= pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])
publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)

# 유명출판사를 조회하여 보면 isbn의 시작이 모두 0140으로 시작하는 것을 확인할 수 있습니다.
# 그러나 표기 방법의 차이 및 오타로 인해 같은 그룹으로 묶이지 못하는 항목이 있습니다.
books[books['isbn'].apply(lambda x: x[:4])=='0140']['publisher'].unique()

# 출판사 등장 횟수가 1보다 큰 출판사들을 모아 리스트로 저장합니다.
modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values

for publisher in modify_list:
    try:
	#현재 출판사에 대한 ISBN의 처음 네 자리를 확인하고, 이들 중에서 가장 많이 등장한 ISBN의 처음 네 자리를 선택합니다.
        number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
        # 선택한 ISBN의 처음 네 자리에 대응되는 출판사 중 가장 많이 등장한 출판사를 찾습니다.
        right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
        # 선택한 ISBN 번호의 출판사들을 가장 많이 등장한 출판사로 통일합니다.
        books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher 
    except:
        pass


# 성능 확인
eda_prograss(users,books)

### data save
users = users.replace(np.nan, "n/a")

# location_city, location_state, location_country 열을 합쳐서 location 열 생성
users['location'] = users['location_city'] + ', ' + users['location_state'] + ', ' + users['location_country']

# 필요 없어진 열 삭제
users.drop(['location_city', 'location_state', 'location_country'], axis=1, inplace=True)


# 기존 CSV 파일에 대체
users.to_csv(data_path + 'users.csv', index=False)
books.to_csv(data_path + 'books.csv', index=False)