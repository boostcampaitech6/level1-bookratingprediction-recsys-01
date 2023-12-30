import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.model_selection import KFold

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6

# 메인 전처리 함수
def process_context_data(users, books, ratings1, ratings2):
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    ratings1 : pd.DataFrame
        train 데이터의 rating
    ratings2 : pd.DataFrame
        test 데이터의 rating
    ----------
    """

    #### CONTEXT

    ## location
    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.drop(['location'], axis=1)

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')

    # user 파트 인덱싱
    loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}

    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
    test_df['age'] = test_df['age'].apply(age_map)

    # book 파트 인덱싱
    category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}

    train_df['category'] = train_df['category'].map(category2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    test_df['category'] = test_df['category'].map(category2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)


    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
    }

    return idx, train_df, test_df


def image_vector(path):
    """
    Parameters
    ----------
    path : str
        이미지가 존재하는 경로를 입력합니다.
    ----------
    """
    img = Image.open(path)
    scale = transforms.Resize((32, 32)) # 이거 더 큰 사이즈로 바꿔도 되겠네
    tensor = transforms.ToTensor()
    img_fe = Variable(tensor(scale(img)))
    return img_fe


def process_img_data(df, books, user2idx, isbn2idx, train=False):
    """
    Parameters
    ----------
    df : pd.DataFrame
        기준이 되는 데이터 프레임을 입력합니다.
    books : pd.DataFrame
        책 정보에 대한 데이터 프레임을 입력합니다.
    user2idx : Dict
        각 유저에 대한 index 정보가 있는 사전을 입력합니다.
    isbn2idx : Dict
        각 책에 대한 index 정보가 있는 사전을 입력합니다.
    ----------
    """
    books_ = books.copy()
    # books_['isbn'] = books_['isbn'].map(isbn2idx)

    if train == True:
        df_ = df.copy()
    else:
        df_ = df.copy()
        # df_['user_id'] = df_['user_id'].map(user2idx)
        # df_['isbn'] = df_['isbn'].map(isbn2idx)

    # book마다 image url 붙이기
    df_ = pd.merge(df_, books_[['isbn', 'img_path']], on='isbn', how='left')
    df_['img_path'] = df_['img_path'].apply(lambda x: 'data/'+x)
    img_vector_df = df_[['img_path']].drop_duplicates().reset_index(drop=True).copy()

    # url에서 이미지 가져오기
    data_box = []
    for idx, path in tqdm(enumerate(sorted(img_vector_df['img_path']))):
        data = image_vector(path)
        if data.size()[0] == 3:
            data_box.append(np.array(data))
        else:
            data_box.append(np.array(data.expand(3, data.size()[1], data.size()[2])))

    # 이미지를 df에 저장
    img_vector_df['img_vector'] = data_box
    df_ = pd.merge(df_, img_vector_df, on='img_path', how='left')
    return df_

def integrated_data_load(args):
    """
    Parameters
    ----------
    Args:
        data_path : str
            데이터 경로
    ----------
    """

    ######################## DATA LOAD
    users = pd.read_csv(args.data_path + 'users.csv')
    books = pd.read_csv(args.data_path + 'books.csv')
    train = pd.read_csv(args.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.data_path + 'sample_submission.csv')


    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

   
    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

     ## image
    img_train = process_img_data(train, books, user2idx, isbn2idx, train=True)
    img_test = process_img_data(test, books, user2idx, isbn2idx, train=False)


    ## context
    idx, context_train, context_test = process_context_data(users, books, train, test)

    ## join
    context_train['img_vector'] = img_train['img_vector']
    context_test['img_vector'] = img_test['img_vector']
    
    field_dims = np.array([ len(user2idx), len(isbn2idx),
                            6, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
                            len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)

    data = {
            'context_train':context_train, # train
            'context_test':context_test, # test
            
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data

class Integrated_Dataset(Dataset):
    def __init__(self, user_isbn_vector, img_vector, label):
        """
        Parameters
        ----------
        user_isbn_vector : np.ndarray
            벡터화된 유저와 책 데이터를 입렵합니다.
        img_vector : np.ndarray
            벡터화된 이미지 데이터를 입력합니다.
        label : np.ndarray
            정답 데이터를 입력합니다.
        ----------
        """
        self.user_isbn_vector = user_isbn_vector # context
        self.img_vector = img_vector # image
        self.label = label # target
        
    def __len__(self):
        return self.user_isbn_vector.shape[0]
    def __getitem__(self, i):
        return {
                'user_isbn_vector' : torch.tensor(self.user_isbn_vector[i], dtype=torch.long),
                'img_vector' : torch.tensor(self.img_vector[i], dtype=torch.float32),
                'label' : torch.tensor(self.label[i], dtype=torch.float32),
                }


def integrated_data_split(args, data):
    """
    Parameters
    ----------
    Args : argparse.ArgumentParser
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            seed 값을 입력합니다.
    data : Dict
        image_data_load로 부터 전처리가 끝난 데이터가 담긴 사전 형식의 데이터를 입력합니다.
    ----------
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['context_train'].drop(columns=['rating']),
                                                        data['context_train']['rating'],
                                                        test_size=args.test_size,
                                                        random_state=args.seed,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

# context: embedding
# image: CNN

# concat -> FM

# def integrated_data_loader(args, data):
#     """
#     Parameters
#     ----------
#     Args : argparse.ArgumentParser
#         batch_size : int
#             Batch size를 입력합니다.
#     data : Dict
#         X(context, image) Y(target) -> context, image, target 세 덩어리로 Dataset 만듦
#         세덩어리로 된 dataset을 dataloader에 넣어서 -> context, image, target 세 덩어리로 튀어나옴
#     ----------
#     """

#     # X, Y
#     # X(context), X(image), Y
#     train_dataset = Integrated_Dataset(
#                                 data['X_train'].drop(columns=['img_vector']).values, # context
#                                 data['X_train']['img_vector'].values, # image
#                                 data['y_train'].values # target
#                                 )
#     valid_dataset = Integrated_Dataset(
#                                 data['X_valid'].drop(columns=['img_vector']).values,
#                                 data['X_valid']['img_vector'].values,
#                                 data['y_valid'].values
#                                 )
#     test_dataset = Integrated_Dataset(
#                                 data['context_test'].drop(columns=['img_vector','rating']).values,
#                                 data['context_test']['img_vector'].values,
#                                 data['context_test']['rating'].values
#                                 )


#     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
#     valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
#     test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)
#     data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader
#     return data

##### k-fold 사용할 때 loader #####
def integrated_data_loader(args, data, k_folds=5):
    """
    K-Fold 교차 검증을 위한 데이터 로더를 생성

    Parameters:
    args (argparse.Namespace): 설정 파라미터가 포함된 객체
    data (dict): 전처리된 데이터가 포함된 딕셔너리
    k_folds (int): 사용할 폴드의 수

    Returns:
    data (dict): 각 K-폴드 데이터 로더가 추가된 딕셔너리
    """
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=args.seed)

    # 각 폴드에 대한 데이터 로더 저장을 위한 딕셔너리
    data['kfold_dataloaders'] = []

    for fold, (train_index, valid_index) in enumerate(kfold.split(data['context_train'])):
        # 학습 및 검증 데이터 분할
        X_train_fold = data['context_train'].iloc[train_index].drop(columns=['img_vector', 'rating'])
        y_train_fold = data['context_train'].iloc[train_index]['rating']
        X_valid_fold = data['context_train'].iloc[valid_index].drop(columns=['img_vector', 'rating'])
        y_valid_fold = data['context_train'].iloc[valid_index]['rating']

        # 이미지 벡터 처리
        img_vector_train = data['context_train'].iloc[train_index]['img_vector'].values
        img_vector_valid = data['context_train'].iloc[valid_index]['img_vector'].values

        # 데이터셋 생성
        train_dataset = Integrated_Dataset(X_train_fold.values, img_vector_train, y_train_fold.values)
        valid_dataset = Integrated_Dataset(X_valid_fold.values, img_vector_valid, y_valid_fold.values)

        # 데이터 로더 생성
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

        # 폴드별 데이터 로더 저장
        data['kfold_dataloaders'].append((train_dataloader, valid_dataloader))
        test_dataset = Integrated_Dataset(
                                data['context_test'].drop(columns=['img_vector','rating']).values,
                                data['context_test']['img_vector'].values,
                                data['context_test']['rating'].values
                                )
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)
        data['test_dataloader'] = test_dataloader

    return data
    # X(context), X(image), Y
    