import numpy as np
import torch
import torch.nn as nn



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

                                        # nn.Conv2d(3, 12, kernel_size=3, stride=2, padding=1),
                                        # nn.ReLU(),
                                        # nn.MaxPool2d(kernel_size=3, stride=2),

                                        # nn.Conv2d(12,3, kernel_size=1),
                                        # nn.ReLU(),

                                        # nn.Conv2d(3, 18, kernel_size=3, stride=2, padding=1),
                                        # nn.ReLU(),
                                        # nn.MaxPool2d(kernel_size=3, stride=2),
                                        
                                        nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1), # [b, 3, 32, 32] -> [b, 6, 32, 32]
                                        nn.ReLU(), # activation
                                        nn.Conv2d(6, 9, kernel_size=3, stride=2, padding=1), # [b, 6, 32, 32] -> [b, 9, 16, 16]
                                        nn.ReLU(), # activation
                                        nn.MaxPool2d(kernel_size=3, stride=2), # [b, 9, 16, 16] -> [b, 9, 7, 7]
                                        nn.Conv2d(9, 12, kernel_size=3, stride=2, padding=1), # [b, 9, 7, 7] -> [b, 12, 4, 4]
                                        nn.ReLU(), # activation
                                        nn.MaxPool2d(kernel_size=3, stride=2), # [b, 12, 4, 4] -> [b, 12, 1, 1]
        )
    def forward(self, x):
        x = self.cnn_layer(x)
        # x = x.view(-1, 12 * 1 * 1) # 12 channel이니 각 채널은 flatten
        x = x.view(-1, 12 * 1 * 1)
        return x




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
    # FM과 동일한 field별 embedding 실행


# cross product -> 차원 안바뀜
class CrossNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])


    def forward(self, x: torch.Tensor):
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x
    # x0는 계속 곱해지는거 / xw는 층별로 전파되는 교호작용


# MLP을 구현합니다.
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()

        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.LeakyReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim

        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)


    def forward(self, x):
        return self.mlp(x)


# Crossnetwork 결과를 MLP layer에 넣어 최종결과를 도출합니다.
class DeepCrossNetworkModel(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']

        self.cnn = CNN_Base()
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim) # 하나의 field
        self.embed_output_dim = len(self.field_dims) * args.embed_dim + 12 # 전체 embedding된 x의 dimension + 12
        self.cn = CrossNetwork(self.embed_output_dim - 12, args.num_layers)


        self.mlp_stacked = MultiLayerPerceptron(self.embed_output_dim, args.mlp_dims, args.dropout, output_layer=False)
        self.mlp_parallel = MultiLayerPerceptron(self.embed_output_dim, args.mlp_dims, args.dropout, output_layer=False)

        self.cd_linear_stacked = nn.Linear(args.mlp_dims[0], 1, bias=False)
        self.cd_linear_parallel1 = nn.Linear(self.embed_output_dim-12 + args.mlp_dims[-1], (self.embed_output_dim-12 + args.mlp_dims[-1])//3, bias=True)
        self.cd_linear_parallel2 = nn.Linear((self.embed_output_dim-12 + args.mlp_dims[-1])//3, 1, bias=True)
        self.cd_linear_parallel = nn.Linear(self.embed_output_dim-12 + args.mlp_dims[-1], 1, bias=True)

        self.bn = nn.BatchNorm1d((self.embed_output_dim-12 + args.mlp_dims[-1])//3)
        self.dp = nn.Dropout(p=0.2)
        

        # self.weights = torch.load('saved_models/1x1_CNN_FM_baseline.pt') # id,isbn CNN
        
        # selected_layers = ['cnn.cnn_layer.0.weight', 'cnn.cnn_layer.0.bias','nn.cnn_layer.3.weight','cnn.cnn_layer.3.bias','cnn.cnn_layer.5.weight','cnn.cnn_layer.5.bias']
        # new_state_dict = {k[4:]: v for k, v in self.weights.items() if k in selected_layers}
        # self.cnn.load_state_dict(new_state_dict, strict=False)
        
        # # 선택된 레이어만 freeze
        # for name, param in self.cnn.named_parameters():
        #     if name in selected_layers:
        #         param.requires_grad = False
        
        # self.cnn.eval()

    # parallel DCN
    def forward(self,x: torch.Tensor):
        user_isbn_vector, img_vector = x[0], x[1] # 앞 두 덩어리: context, image

        # 1) IMAGE
        img_feature = self.cnn(img_vector) # img -> embedding

        # 2) CONTEXt
        embed_x = self.embedding(user_isbn_vector)
        embed_x = embed_x.view(-1, embed_x.size(1) * embed_x.size(2))

        # 3)-2 cross
        x_l1 = self.cn(embed_x) # cross

        # 3)-1 CONCAT for mlp
        feature_embedding = torch.cat([
                                    embed_x,
                                    img_feature
                                    ], dim=1)
        x_l2 = self.mlp_parallel(feature_embedding) # mlp
        

        # 3) OUTPUT LAYER
        x_out = torch.concat([x_l1, x_l2], dim=1)

        p1 = self.cd_linear_parallel(x_out)

        return p1.squeeze(1)


