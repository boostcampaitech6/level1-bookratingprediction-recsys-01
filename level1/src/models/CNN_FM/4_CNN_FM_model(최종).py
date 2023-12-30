import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

## context: embedding
## image: CNN

# concat -> linear -> FM

# factorization을 통해 얻은 feature를 embedding 합니다.
class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)
        # 전체 field들의 one-hot -> embedding



# FM 모델을 구현합니다.
class FactorizationMachineModel(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        self.linear = FeaturesLinear(self.field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)


    def forward(self, x: torch.Tensor):
        x = self.linear(x) + self.fm(self.embedding(x))
        # return torch.sigmoid(x.squeeze(1))
        return x.squeeze(1)



# feature 사이의 상호작용을 효율적으로 계산합니다.
class FactorizationMachine(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.v = nn.Parameter(torch.rand(input_dim, latent_dim), requires_grad = True)

        self.linear = nn.Linear(98,1, bias=True)
        
        self.bn1 = nn.BatchNorm1d(30)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(98, 30, bias=False)
        self.linear2 = nn.Linear(30,1)
        ## 수동
        # emb_dim=8 -> 92 / n -> n*10+12


    def forward(self, x):
        x1 = self.linear1(x) #FM:FeaturesLinear
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.dp(x1)

        mlp = self.linear2(x1)

        linear = self.linear(x)

        square_of_sum = torch.mm(x, self.v) ** 2
        sum_of_square = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True) #FM:FactorizationMachine
        output = linear + mlp #+ (0.5 * pair_interactions)
        return output


# 이미지 특징 추출을 위한 기초적인 CNN Layer를 정의합니다.
class CNN_Base(nn.Module):
    def __init__(self, ):
        super(CNN_Base, self).__init__()
        self.cnn_layer = nn.Sequential(
                                        # nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1),
                                        # nn.ReLU(),
                                        # nn.MaxPool2d(kernel_size=3, stride=2),
                                        # nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1),
                                        # nn.ReLU(),
                                        # nn.MaxPool2d(kernel_size=3, stride=2),

                                        nn.Conv2d(3, 12, kernel_size=3, stride=2, padding=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3, stride=2),

                                        nn.Conv2d(12,3, kernel_size=1),
                                        nn.ReLU(),

                                        nn.Conv2d(3, 18, kernel_size=3, stride=2, padding=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3, stride=2),
                                        )
    def forward(self, x):
        x = self.cnn_layer(x)
        # x = x.view(-1, 12 * 1 * 1) # 12 channel이니 각 채널은 flatten
        x = x.view(-1, 18 * 1 * 1)
        return x



# 기존 유저/상품 벡터와 이미지 벡터를 결합하여 FM으로 학습하는 모델을 구현합니다.
class CNN_FM(torch.nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims'] # context embedding dims
        self.embedding = FeaturesEmbedding(self.field_dims, args.cnn_embed_dim) # cnn_embd_dim=64 (main.py에서 설정 가능)
        self.cnn = CNN_Base()

        # load & freeze
        # self.weights = torch.load('saved_models/1x1_CNN_FM_baseline.pt') # id,isbn CNN
        
        # selected_layers = ['cnn.cnn_layer.0.weight', 'cnn.cnn_layer.0.bias','nn.cnn_layer.3.weight','cnn.cnn_layer.3.bias','cnn.cnn_layer.5.weight','cnn.cnn_layer.5.bias']
        # new_state_dict = {k[4:]: v for k, v in self.weights.items() if k in selected_layers}
        # self.cnn.load_state_dict(new_state_dict, strict=False)

        # # 선택된 레이어만 freeze
        # for name, param in self.cnn.named_parameters():
        #     if name in selected_layers:
        #         param.requires_grad = False
        # self.cnn.eval()

        self.fm = FactorizationMachine( #arg.cnn_embed_dim*(context의 field 수) + 12
                                        # input_dim=(args.cnn_embed_dim * len(self.field_dims)+12 * 1 * 1),
                                        input_dim=(args.cnn_embed_dim * len(self.field_dims)+18 * 1 * 1),
                                        latent_dim=args.cnn_latent_dim, #v의 dimension
                                        )


    def forward(self, x):
        user_isbn_vector, img_vector = x[0], x[1] # 앞 두 덩어리: context, image
        user_isbn_feature = self.embedding(user_isbn_vector) # context -> embedding
        img_feature = self.cnn(img_vector) # img -> embedding

        feature_vector = torch.cat([
                                    user_isbn_feature.view(-1, user_isbn_feature.size(1) * user_isbn_feature.size(2)),
                                    img_feature
                                    ], dim=1)
        
        # linear에 embedding (vi,vj)
        # fm에 embedding
        output = self.fm(feature_vector)
        #output = self.linear(feature_vector) + self.fm(feature_vector)

        return output.squeeze(1)

# FM
# user : id : age, location
# book : isbn: [year, category, language], [image]

# DCN

# CNN_FM
# (user_id <-> book_isbn) => 2.3
# 맥락