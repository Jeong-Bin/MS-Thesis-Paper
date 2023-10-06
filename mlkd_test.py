import os
from datetime import datetime
import torch
from torch import nn
import learn2learn as l2l
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import timm
from models.BaseModels import CNN4, CNN1
from utils import seed_fixer, index_preprocessing, knowledge_distillation_loss, confidence_interval, BaselineErrors
from data.DataPreprocessing import make_df, CustomDataset, Meta_Transforms

import warnings
warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=2023, type=int)
parser.add_argument("--dataset_name", default='mini_imagenet', type=str,
                    choices=['mini_imagenet', 'CUB_200_2011', 'CIFAR_FS', 'Aircraft'])

parser.add_argument("--save_dir", default='saved_models_oc', type=str)

parser.add_argument("--test_way", default=5, type=int)
parser.add_argument("--test_shot", default=1, type=int)
parser.add_argument("--test_query", default=15, type=int)

parser.add_argument("--adapt_lr", default=0.01, type=float)
parser.add_argument("--train_adapt_steps", default=5, type=int)
parser.add_argument("--test_adapt_steps", default=5, type=int)

parser.add_argument("--task_batch_size", default=2000, type=int)

parser.add_argument("--meta_wrapper", default='maml', type=str)
parser.add_argument("--first_order", default=False, type=bool)

parser.add_argument("--kd_T", default=2, type=float)
parser.add_argument("--kd_mode", default='dual', type=str,
                    choices=['inner', 'outer', 'dual'])
parser.add_argument("--lambda_inner_ce", default=1.0, type=float)
parser.add_argument("--lambda_inner_kd", default=1.0, type=float)

parser.add_argument("--teacher_backbone", default='convnext_small.in12k_ft_in1k_384', type=str,
                    choices=['efficientnet_b1', 'wide_resnet50_2', 'convnext_small.in12k_ft_in1k_384'])

parser.add_argument("--device", default='cuda', type=str)

parser.add_argument("--teacher_saved_model", default='', type=str)
parser.add_argument("--student_saved_model", default='', type=str)

args = parser.parse_args()
parser.add_argument("--test_data_dir", default=f'/home/hjb3880/WORKPLACE/datasets/{args.dataset_name}_C', type=str)
args = parser.parse_args()
config = {
        'argparse' : args,
        'save_name_tag' : f'', ################################################
        'memo' : '' 
}

if args.device == 'cuda' :
    device = torch.device('cuda')
elif args.device == 'cpu' :
    device = torch.device('cpu')


if args.dataset_name in ['MiniImagenet', 'mini_imagenet', 'mini-imagenet'] :
    dname = 'mini'
elif args.dataset_name in ['cub', 'cub_200_2011', 'CUB', 'CUB_200_2011'] :
    dname = 'cub'
elif args.dataset_name in ['TieredImagenet', 'tiered_imagenet', 'tiered-imagenet'] :
    dname = 'tiered'
elif args.dataset_name in ['FC100', 'fc100'] :
    dname = 'fc100'
elif args.dataset_name in ['cifar_fs', 'cifar-fs', 'CIFAR_FS', 'CIFAR-FS'] :
    dname = 'cifar'
elif args.dataset_name in ['Aircraft', 'aircraft'] :
    dname = 'airc'

save_name = f"{dname}_{args.test_way}w{args.test_shot}s_{args.kd_mode}KD_f{args.first_order}" 

import wandb
run = wandb.init(project="TEST_OC")
wandb.run.name = save_name
wandb.run.save()
wandb.config.update(config)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, adaptation_indices, evaluation_indices, 
               teacher_backbone, teacher_learner, student_learner, 
               criterion, adaptation_steps, device):
    
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # teacher's data
    feature = teacher_backbone.forward_features(data)
    adaptation_feature = feature[adaptation_indices]

    # student's data
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices] # support set
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices] # query set

    # Inner loop
    for step in range(adaptation_steps):
        teacher_adapt_logit = teacher_learner(adaptation_feature)
        teacher_adapt_error = criterion(teacher_adapt_logit, adaptation_labels)
        teacher_learner.adapt(teacher_adapt_error)

        student_adapt_logit = student_learner(adaptation_data)
        student_adapt_error = criterion(student_adapt_logit, adaptation_labels)
        if args.kd_mode in ['inner', 'dual']:
            kd_loss = knowledge_distillation_loss(student_adapt_logit, teacher_adapt_logit, args.kd_T)
            student_adapt_error = args.lambda_inner_ce*student_adapt_error + args.lambda_inner_kd*kd_loss
        student_learner.adapt(student_adapt_error)

    student_eval_logit = student_learner(evaluation_data)
    student_evaluation_error = criterion(student_eval_logit, evaluation_labels)
    student_evaluation_accuracy = accuracy(student_eval_logit, evaluation_labels)

    return student_evaluation_error, student_evaluation_accuracy
    

################################## Test ##################################

now = datetime.now()
print(f"Start time: {now.strftime('%Y-%m-%d %H:%M:%S')}")

def meta_test(args):

    seed_fixer(args.seed)

    test_data_transforms = A.Compose([
        # Already resized in the process of creating the C dataset.
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
        ToTensorV2()
    ])

    test = make_df(root=args.test_data_dir, mode='test')
    test_dataset = CustomDataset(test['img_path'].values, test['label'].values, test_data_transforms)
    print('Making Test Tasksets...')
    test_tasksets = Meta_Transforms(dataset = test_dataset, 
                                    way = args.test_way, 
                                    shot = args.test_shot, 
                                    query = args.test_query, 
                                    num_tasks = -1)
    
    test_adapt_idx, test_eval_idx = index_preprocessing(way=args.test_way, shot=args.test_shot, query=args.test_query)

    if args.dataset_name in ['mini_imagenet', 'CUB_200_2011', 'Aircraft']:
        img_sige = 84
        hidden_dim = 64
        spatial_size = 5
    elif args.dataset_name == 'CIFAR_FS':
        img_sige = 32
        hidden_dim = 32
        spatial_size = 2

    # Teacher
    teacher_backbone = timm.create_model(args.teacher_backbone, features_only=False, pretrained=True, num_classes=args.test_way)
    a = teacher_backbone.forward_features(torch.randn(1,3,img_sige,img_sige))
    teacher_backbone.to(device)

    teacher = CNN1(channel_size=a.size(1), kernel_size=a.size(2), num_classes=args.test_way)
    teacher_maml = l2l.algorithms.MAML(teacher, lr=args.adapt_lr, first_order=False)
    teacher_maml.to(device)

    filename = os.path.join(args.save_dir, f"{args.teacher_saved_model}.pth")
    teacher_maml.load_state_dict(torch.load(filename))

    # Student
    student = CNN4(hidden_dim=hidden_dim, spatial_size=spatial_size, num_classes=args.test_way)
    if args.meta_wrapper == 'maml' :
        student_maml = l2l.algorithms.MAML(student, lr=args.adapt_lr, first_order=args.first_order)
    elif args.meta_wrapper == 'metasgd' :
        student_maml = l2l.algorithms.MetaSGD(student, lr=0.01, first_order=args.first_order)
        args.train_adapt_steps = 1
        args.test_adapt_steps = 1
    student_maml.to(device)

    filename = os.path.join(args.save_dir, f"{args.student_saved_model}.pth")
    student_maml.load_state_dict(torch.load(filename))

    criterion = nn.CrossEntropyLoss(reduction='mean')

    student_accuracy_list = []
    student_loss_list = []
    for task in tqdm(range(1, args.task_batch_size+1)):
        teacher_learner = teacher_maml.clone()
        student_learner = student_maml.clone()
        test_batch = test_tasksets.sample()
        student_loss, student_accuracy = fast_adapt(test_batch,
                                                    test_adapt_idx, 
                                                    test_eval_idx,
                                                    teacher_backbone,
                                                    teacher_learner,
                                                    student_learner,
                                                    criterion,
                                                    args.test_adapt_steps,
                                                    device = device,
                                                    )
        student_accuracy_list.append(student_accuracy.item()*100)
        student_loss_list.append(student_loss.item())

        if task <= 10 :
            print(f"[{task}/{args.task_batch_size}] acc:{student_accuracy*100:.3f}, loss:{student_loss:.4f}")
        
        if task%1000 == 0 :
            ci = confidence_interval(student_accuracy_list)
            test_student_accuracy = sum(student_accuracy_list) /task
            test_student_loss = sum(student_loss_list) /task
            print('.\n.\n.',)
            print(f"Test {task}")
            print(f"Test Accuracy (95% ci) : {test_student_accuracy:.2f} ±{ci['95%']:.2f}" )
            print(f"Test Loss : {test_student_loss:.4f}")
        

if __name__ == '__main__':
    meta_test(args)

now = datetime.now()
print(f"End time: {now.strftime('%Y-%m-%d %H:%M:%S')}")





# def meta_test(args):
#     seed_fixer(args.seed)

#     if args.dataset_name in ['mini_imagenet', 'CUB_200_2011']:
#         img_sige = 84
#         hidden_dim = 64
#         spatial_size = 5
#     elif args.dataset_name == 'CIFAR_FS':
#         img_sige = 32
#         hidden_dim = 32
#         spatial_size = 2

#     # Teacher
#     teacher_backbone = timm.create_model(args.teacher_backbone, features_only=False, pretrained=True, num_classes=args.test_way)
#     a = teacher_backbone.forward_features(torch.randn(1,3,img_sige,img_sige))
#     teacher_backbone.to(device)

#     teacher = CNN1(channel_size=a.size(1), kernel_size=a.size(2), num_classes=args.test_way)
#     teacher_maml = l2l.algorithms.MAML(teacher, lr=args.adapt_lr, first_order=False)
#     teacher_maml.to(device)

#     filename = os.path.join(args.save_dir, f"{args.teacher_saved_model}.pth")
#     teacher_maml.load_state_dict(torch.load(filename))

#     # Student
#     student = CNN4(hidden_dim=hidden_dim, spatial_size=spatial_size, num_classes=args.test_way)
#     if args.meta_wrapper == 'maml' :
#         student_maml = l2l.algorithms.MAML(student, lr=args.adapt_lr, first_order=args.first_order)
#     elif args.meta_wrapper == 'metasgd' :
#         student_maml = l2l.algorithms.MetaSGD(student, lr=0.01, first_order=args.first_order)
#         args.train_adapt_steps = 1
#         args.test_adapt_steps = 1
#     student_maml.to(device)

#     filename = os.path.join(args.save_dir, f"{args.student_saved_model}.pth")
#     student_maml.load_state_dict(torch.load(filename))

#     criterion = nn.CrossEntropyLoss(reduction='mean')

    
#     test_data_transforms = A.Compose([
#         # Already resized in the process of creating the C dataset.
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
#         ToTensorV2()
#     ])


#     baseline_sum_error = BaselineErrors()
#     mce = 0
#     test_adapt_idx, test_eval_idx = index_preprocessing(way=args.test_way, shot=args.test_shot, query=args.test_query)
#     for c in baseline_sum_error.keys():
#         test_student_error_all = []
#         for i in range(1, 6):
#             test = make_df(root=args.test_data_dir, mode='test', corruption=c, level=i)
#             test_dataset = CustomDataset(test['img_path'].values, test['label'].values, test_data_transforms)
#             test_tasksets = Meta_Transforms(dataset = test_dataset, 
#                                             way = args.test_way, 
#                                             shot = args.test_shot, 
#                                             query = args.test_query, 
#                                             num_tasks = -1)
            
#             student_error_list = []
#             for task in tqdm(range(1, args.task_batch_size+1)):
#                 teacher_learner = teacher_maml.clone()
#                 student_learner = student_maml.clone()
#                 test_batch = test_tasksets.sample()
#                 student_loss, student_accuracy = fast_adapt(test_batch,
#                                                             test_adapt_idx, 
#                                                             test_eval_idx,
#                                                             teacher_backbone,
#                                                             teacher_learner,
#                                                             student_learner,
#                                                             criterion,
#                                                             args.test_adapt_steps,
#                                                             device = device,
#                                                             )
#                 student_error_list.append(1-student_accuracy.item())

                
#             test_student_error = sum(student_error_list) /task
#             test_student_error_all.append(test_student_error)
#         ce = sum(test_student_error_all) / baseline_sum_error[c]
#         print(f'crrouption: {c}') 
#         print(f'CE: {ce}')
#         mce += ce
#     print()
#     print(f'mCE: {mce / len(baseline_sum_error)}')

        

# if __name__ == '__main__':
#     meta_test(args)

# now = datetime.now()
# print(f"End time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
