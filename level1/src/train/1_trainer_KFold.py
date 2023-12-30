import os
import tqdm
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import SGD, Adam
from sklearn.model_selection import KFold
import copy
from torch.optim.lr_scheduler import StepLR

import pandas as pd

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6
    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss


# def train(args, model, dataloader, logger, setting):
#     # torch.cuda.empty_cache()
#     # torch.cuda.empty_cache()
#     minimum_loss = 999999999
#     fold_losses = []

#     if args.loss_fn == 'MSE':
#         loss_fn = MSELoss()
#     elif args.loss_fn == 'RMSE':
#         loss_fn = RMSELoss()
#     else:
#         pass
#     if args.optimizer == 'SGD':
#         optimizer = SGD(model.parameters(), lr=args.lr)
#     elif args.optimizer == 'ADAM':
#         optimizer = Adam(model.parameters(), lr=args.lr)
#     else:
#         pass
#     for vaild_fold,item in enumerate(dataloader['kfold_dataloaders']):   #(index, [(),(),(),(),()])
#         train_data,valid_data = item     #item -> list[(loader,loader),(tuple),(tuple),(tuple)]
#         for epoch in tqdm.tqdm(range(args.epochs)):
#             model.train()
#             total_loss = 0
#             batch = 0
#             for idx, data in enumerate(train_data):
#                 if args.model == 'CNN_FM':
#                     x, y = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
#                 elif args.model == 'DeepCoNN':
#                     x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
#                 else:
#                     x, y = data[0].to(args.device), data[1].to(args.device)
#                 y_hat = model(x)
#                 loss = loss_fn(y.float(), y_hat)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()
#                 batch +=1
                
#             valid_loss = valid(args, model, valid_data, loss_fn)
#             print(f'FOLD: {vaild_fold}, Epoch: {epoch+1}, Train_loss: {total_loss/batch:.3f}, valid_loss: {valid_loss:.3f}')
#             logger.log(epoch=epoch+1, train_loss=total_loss/batch, valid_loss=valid_loss)
#             if minimum_loss > valid_loss:
#                 minimum_loss = valid_loss
#                 os.makedirs(args.saved_model_path, exist_ok=True)
#                 torch.save(model.state_dict(), f'{args.saved_model_path}/{setting.save_time}_{args.model}_model_fold{vaild_fold}.pt')
#     logger.close()
#     return model
def train(args, model, dataloader, logger, setting):
    minimum_loss = 999999999
    if args.loss_fn == 'MSE':
        loss_fn = MSELoss()
    elif args.loss_fn == 'RMSE':
        loss_fn = RMSELoss()
    else:
        pass
    if args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'ADAM':
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        pass

    for epoch in tqdm.tqdm(range(args.epochs)):
        model.train()
        total_loss = 0
        batch = 0

        for idx, data in enumerate(dataloader['train_dataloader']):
            if args.model in ('CNN_FM','CNN_contextFM','DCN'):
                x, y = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
            elif args.model == 'DeepCoNN':
                x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
            else:
                x, y = data[0].to(args.device), data[1].to(args.device)
            y_hat = model(x)
            loss = loss_fn(y.float(), y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch +=1
        
        valid_loss = valid(args, model, dataloader, loss_fn)
        print(f'Epoch: {epoch+1}, Train_loss: {total_loss/batch:.3f}, valid_loss: {valid_loss:.3f}')
        logger.log(epoch=epoch+1, train_loss=total_loss/batch, valid_loss=valid_loss)
        if minimum_loss > valid_loss:
            minimum_loss = valid_loss
            os.makedirs(args.saved_model_path, exist_ok=True)
            torch.save(model.state_dict(), f'{args.saved_model_path}/{setting.save_time}_{args.model}_model.pt')
    logger.close()
    return model


# def valid(args, model, dataloader, loss_fn):
#     model.eval()
#     total_loss = 0
#     batch = 0

#     for idx, data in enumerate(dataloader['valid_dataloader']):
#         if args.model in ('CNN_FM','CNN_contextFM','DCN'):
#             x, y = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
#         elif args.model == 'DeepCoNN':
#             x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
#         else:
#             x, y = data[0].to(args.device), data[1].to(args.device)
#         y_hat = model(x)
#         loss = loss_fn(y.float(), y_hat)
#         total_loss += loss.item()
#         batch +=1


#     valid_loss = total_loss/batch
#     return valid_loss

def valid(args, model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    batch = 0
    for idx, data in enumerate(dataloader):
        if args.model in ('CNN_FM', 'DCN'):
            x, y = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
        elif args.model == 'DeepCoNN':
            x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
        else:
            x, y = data[0].to(args.device), data[1].to(args.device)
        y_hat = model(x)
        loss = loss_fn(y.float(), y_hat)
        total_loss += loss.item()
        batch +=1
    valid_loss = total_loss/batch
    return valid_loss


def test(args, model, dataloader, setting):
    predicts = list()
    if args.use_best_model == True:
        model.load_state_dict(torch.load(f'./saved_models/{setting.save_time}_{args.model}_1fold_1epoch_model.pt'))
    else:
        pass
    model.eval()

    for idx, data in enumerate(dataloader['test_dataloader']):
        
        if args.model in ('CNN_FM','CNN_contextFM','DCN'):
            x, _ = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
        elif args.model == 'DeepCoNN':
            x, _ = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
        else:
            x = data[0].to(args.device)
        y_hat = model(x)
        predicts.extend(y_hat.tolist())
    return predicts

# 1) test loader에 왜 전체 데이터가 들어가있는가?
# 2) validation data를 학습한 것이 아닌가?
# 3) rating이 혹시 학습 데이터에 포함되지는 않았는가?

def train1(args, model, dataloader, logger, setting):
    minimum_loss = 999999999
    
    for valid_fold,item in enumerate(dataloader['kfold_dataloaders']):   #(index, [(),(),(),(),()])
        train_data,valid_data = item
        model_tmp=copy.deepcopy(model)

        
        if args.loss_fn == 'MSE':
            loss_fn = MSELoss()
        elif args.loss_fn == 'RMSE':
            loss_fn = RMSELoss()
        else:
            pass
        if args.optimizer == 'SGD':
            optimizer = SGD(model_tmp.parameters(), lr=args.lr)
        elif args.optimizer == 'ADAM':
            optimizer = Adam(model_tmp.parameters(), lr=args.lr)
        else:
            pass

        scheduler = StepLR(optimizer, step_size=3, gamma=0.37)   #스케쥴러 추가

        for epoch in tqdm.tqdm(range(args.epochs)):
            model_tmp.train()
            total_loss = 0
            batch = 0
            for idx, data in enumerate(train_data):
                if args.model in ('CNN_FM', 'DCN'):
                    x, y = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
                elif args.model_tmp == 'DeepCoNN':
                    x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
                else:
                    x, y = data[0].to(args.device), data[1].to(args.device)
                y_hat = model_tmp(x)
                loss = loss_fn(y.float(), y_hat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch +=1
            scheduler.step()     # 스케쥴러 추가
            valid_loss = valid(args, model_tmp, valid_data, loss_fn)
            print(f'FOLD: {valid_fold+1}, Epoch: {epoch+1}, Train_loss: {total_loss/batch:.3f}, valid_loss: {valid_loss:.3f}')
            logger.log(epoch=epoch+1, train_loss=total_loss/batch, valid_loss=valid_loss)

            ######  이렇게 하면 K_fold에서 k=1,2,3,4,5 중에서 가장 좋은 성능의 모델 하나가 저장됨  #####
            # if minimum_loss > valid_loss:
            #     minimum_loss = valid_loss
            os.makedirs(args.saved_model_path, exist_ok=True)
            torch.save(model_tmp.state_dict(), f'{args.saved_model_path}/{setting.save_time}_{args.model}_{valid_fold+1}fold_{epoch+1}epoch_model.pt')  # 덮어씌워지겠지..

            print(f'--------------- {args.model} PREDICT ---------------')
            predicts = test(args, model_tmp, dataloader, setting)


            ######################## SAVE PREDICT
            print(f'--------------- SAVE {args.model} PREDICT ---------------')
            submission = pd.read_csv(args.data_path + 'sample_submission.csv')
            if args.model in ('FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN', 'CNN_context_FM'):
                submission['rating'] = predicts
            else:
                pass

            # filename = setting.get_submit_filename(args)
            submission.to_csv(f'submit/{setting.save_time}_{args.model}_{valid_fold+1}fold_{epoch+1}epoch.csv', index=False)

    logger.close()
    return model_tmp

##### k-fold에서 나온 성능이 가장 좋은 모델들의 parameter 평균낸 모델 반환 #####
def average_weights(weights_list):
    avg_weights = {}
    for key in weights_list[0].keys():
        avg_weights[key] = sum(weight[key] for weight in weights_list) / len(weights_list)
    return avg_weights

def train2(args, model, dataloader, logger, setting):
    model_weights=[]
    minimum_loss = 999999999

    for vaild_fold,item in enumerate(dataloader['kfold_dataloaders']):   #(index, [(),(),(),(),()])
        train_data,valid_data = item
        model_tmp=copy.deepcopy(model)

        if args.loss_fn == 'MSE':
            loss_fn = MSELoss()
        elif args.loss_fn == 'RMSE':
            loss_fn = RMSELoss()
        else:
            pass
        if args.optimizer == 'SGD':
            optimizer = SGD(model_tmp.parameters(), lr=args.lr)
        elif args.optimizer == 'ADAM':
            optimizer = Adam(model_tmp.parameters(), lr=args.lr)
        else:
            pass

        scheduler = StepLR(optimizer, step_size=2, gamma=0.3)
        for epoch in tqdm.tqdm(range(args.epochs)):
            model_tmp.train()
            total_loss = 0
            batch = 0
            for idx, data in enumerate(train_data):
                if args.model in ('CNN_FM', 'DCN'):
                    x, y = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
                elif args.model_tmp == 'DeepCoNN':
                    x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
                else:
                    x, y = data[0].to(args.device), data[1].to(args.device)
                y_hat = model_tmp(x)
                loss = loss_fn(y.float(), y_hat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch +=1
            scheduler.step()     # 스케쥴러 추가
            valid_loss = valid(args, model_tmp, valid_data, loss_fn)
            print(f'FOLD: {vaild_fold+1}, Epoch: {epoch+1}, Train_loss: {total_loss/batch:.3f}, valid_loss: {valid_loss:.3f}')
            logger.log(epoch=epoch+1, train_loss=total_loss/batch, valid_loss=valid_loss)

            if minimum_loss > valid_loss:
                minimum_loss = valid_loss
                model_weight = copy.deepcopy(model_tmp.state_dict())
            
        model_weights.append(model_weight)
    
    average_weight = average_weights(model_weights)
    average_model = copy.deepcopy(model)
    average_model.load_state_dict(average_weight)

    ##### 평균낸 모델 저장 & 저장경로 설정 #####
    os.makedirs(args.saved_model_path, exist_ok=True)
    torch.save(average_model.state_dict(), f'{args.saved_model_path}/{setting.save_time}_{args.model}_model.pt')

    logger.close()
    return average_model