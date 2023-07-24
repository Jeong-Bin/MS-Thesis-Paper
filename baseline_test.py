import os
from datetime import datetime

import torch
from torch import nn
import learn2learn as l2l

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from models.base_model import CNN4
from utils.util import seed_fixer, index_preprocessing
from utils.CustomDatasets import make_df, CustomDataset, Meta_Transforms

import warnings
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=2023, type=int)
parser.add_argument("--dataset_name", default='cub', type=str,
                    choices=['MiniImagenet', 'mini_imagenet', 'mini-imagenet',
                             'cub', 'CUB', 'CUB_200_2011',
                            'TieredImagenet', 'tiered_imagenet', 'tiered-imagenet',
                            'FC100', 'fc100', 'CIFARFS', 'cifarfs'])
parser.add_argument("--aug_mode", default='albumentations', type=str,
                    choices=['torchvision', 'transforms',
                             'albumentations', 'Albumentations', 'alb', 'Alb'])

parser.add_argument("--corruption_data_dir", default='/home/hjb3880/WORKPLACE/datasets/CUB_200_2011_C', type=str)
parser.add_argument("--save_dir", default='saved_models_important', type=str)

parser.add_argument("--train_way", default=10, type=int)
parser.add_argument("--train_shot", default=5, type=int)
parser.add_argument("--train_query", default=5, type=int)

parser.add_argument("--test_way", default=10, type=int)
parser.add_argument("--test_shot", default=5, type=int)
parser.add_argument("--test_query", default=15, type=int)

# parser.add_argument("--lr", default=1e-3, type=float)
# parser.add_argument("--scheduler_step", default=50, type=int)
# parser.add_argument("--scheduler_gamma", default=0.9, type=float)

parser.add_argument("--adapt_lr", default=0.01, type=float)
parser.add_argument("--train_adapt_steps", default=5, type=int)
parser.add_argument("--test_adapt_steps", default=10, type=int)

parser.add_argument("--task_batch_size", default=1000, type=int)
# parser.add_argument("--num_iterations", default=5000, type=int)

parser.add_argument("--meta_wrapping", default='maml', type=str)
parser.add_argument("--first_order", default=True, type=bool)

parser.add_argument("--device", default='cuda', type=str)

parser.add_argument("--student_saved_model", default='0717_Train_cub_10w5s_wrn50_outerKD_strong_OC2_10_Student', type=str) #0717_Train_cub_5w1s_wrn50_outerKD_strong_OC2_10_Student

args = parser.parse_args()

config = {
        'argparse' : args,
        'save_name_tag' : f'wrn50_outerKD_strong_OC2_10', #strong#origin##############################################
        'readme' : 'kd_T:10, test_adapt_steps:10, valid_tasksets:-1'
}

if args.device == 'cuda' :
    device = torch.device('cuda')
elif args.device == 'cpu' :
    device = torch.device('cpu')


if args.dataset_name in ['MiniImagenet', 'mini_imagenet', 'mini-imagenet'] :
    dname = 'mini'
elif args.dataset_name in ['cub', 'CUB', 'CUB_200_2011'] :
    dname = 'cub'
elif args.dataset_name in ['TieredImagenet', 'tiered_imagenet', 'tiered-imagenet'] :
    dname = 'tiered'
elif args.dataset_name in ['FC100', 'fc100'] :
    dname = 'fc100'
elif args.dataset_name in ['CIFARFS', 'cifarfs'] :
    dname = 'cifarfs'

save_name = f"Test_{dname}_{args.test_way}w{args.test_shot}s_{config['save_name_tag']}_train{args.train_way}w{args.train_shot}s"

import wandb
run = wandb.init(project="MAML_C_test")
wandb.run.name = save_name
wandb.run.save()
wandb.config.update(config)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

#==========================================================================================================


def fast_adapt(batch, adaptation_indices, evaluation_indices, 
               student_learner, 
               criterion, adaptation_steps,device):
    
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # student's data
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices] # support set
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices] # query set

    # Inner loop
    for step in range(adaptation_steps):
        student_adapt_logit = student_learner(adaptation_data)
        student_adapt_error = criterion(student_adapt_logit, adaptation_labels)
        student_learner.adapt(student_adapt_error)


    student_eval_logit = student_learner(evaluation_data)
    student_evaluation_error = criterion(student_eval_logit, evaluation_labels)
    student_evaluation_accuracy = accuracy(student_eval_logit, evaluation_labels)

    return  student_evaluation_error, student_evaluation_accuracy
    




#==========================================================================================================


################################## Train ##################################


now = datetime.now()
print(f"Start time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
def meta_test(args):
    acc_list = []
    acc_dict = {}
    for seed in range(1, 51):
        seed_fixer(seed)
        print(seed, '='*30)

        test_data_transforms = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensorV2()
        ])


        test = make_df(root=args.corruption_data_dir, mode='test')
        test_dataset = CustomDataset(test['img_path'].values, test['label'].values, test_data_transforms)
        print('Making Test Tasksets...')
        test_tasksets = Meta_Transforms(dataset = test_dataset, 
                                        way = args.test_way, 
                                        shot = args.test_shot, 
                                        query = args.test_query, 
                                        num_tasks = -1)

        test_adapt_idx, test_eval_idx = index_preprocessing(way=args.test_way, shot=args.test_shot, query=args.test_query)


        # maml Model
        student = CNN4(num_classes=args.train_way) 
        if args.meta_wrapping == 'maml' :
            student_maml = l2l.algorithms.MAML(student, lr=args.adapt_lr, first_order=args.first_order)
        elif args.meta_wrapping == 'metasgd' :
            student_maml = l2l.algorithms.MetaSGD(student, lr=0.01, first_order=args.first_order)
            args.train_adapt_steps = 1
            args.test_adapt_steps = 1
        student_maml.to(device)

        filename = os.path.join(args.save_dir, f"{args.student_saved_model}.pth")
        student_maml.load_state_dict(torch.load(filename))

        criterion = nn.CrossEntropyLoss(reduction='mean')


        test_student_loss_sum = 0.0
        test_student_accuracy_sum = 0.0
        for task in range(1, args.task_batch_size+1):
            student_learner = student_maml.clone()
            test_batch = test_tasksets.sample()
            student_loss, student_accuracy = fast_adapt(test_batch,
                                                        test_adapt_idx, 
                                                        test_eval_idx,
                                                        student_learner,
                                                        criterion,
                                                        args.test_adapt_steps,
                                                        device = device,
                                                        )
            test_student_loss_sum += student_loss.item() 
            test_student_accuracy_sum += student_accuracy.item()
            #print(f"[{task}/{args.task_batch_size}] acc:{student_accuracy*100:.3f}, loss:{student_loss:.4f}")

        # Print some metrics
        test_student_accuracy = test_student_accuracy_sum /args.task_batch_size *100
        test_student_loss = test_student_loss_sum /args.task_batch_size
        print(f"Test Accuracy : {test_student_accuracy:.3f}" )
        print(f"Test Loss : {test_student_loss:.4f}")

        test_student_accuracy = round(test_student_accuracy, 3)
        acc_list.append(test_student_accuracy)
        acc_dict[seed] = test_student_accuracy

    print(acc_list)
    print(acc_dict)

if __name__ == '__main__':
    meta_test(args)

now = datetime.now()
print(f"End time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
