import os
from datetime import datetime
import torch
from torch import nn, optim
import learn2learn as l2l
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import timm
from models.BaseModels import CNN4, CNN1
from utils import seed_fixer, index_preprocessing, knowledge_distillation_loss
from data.DataPreprocessing import make_df, CustomDataset, Meta_Transforms

import warnings
warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=2023, type=int)
parser.add_argument("--dataset_name", default='CIFAR_FS', type=str,
                    choices=['mini_imagenet', 'CUB_200_2011', 'CIFAR_FS'])

parser.add_argument("--train_data_dir", default='/home/hjb3880/WORKPLACE/datasets/CIFAR_FS', type=str)
parser.add_argument("--test_data_dir", default='/home/hjb3880/WORKPLACE/datasets/CIFAR_FS_C', type=str)
parser.add_argument("--save_dir", default='saved_models_oc', type=str)

parser.add_argument("--way", default=5, type=int)
parser.add_argument("--train_shot", default=1, type=int)
# parser.add_argument("--train_query", default=1, type=int)
# parser.add_argument("--test_shot", default=1, type=int)
parser.add_argument("--test_query", default=15, type=int)

# parser.add_argument("--task_batch_size", default=10, type=int,
#                     help="""Recommand: 10 and 5 for the 1-shot and 5&10-shot when using strong aug dataset.
#                             Recommand: 4 and 2 for the 1-shot and 5&10-shot when using clean dataset.
#                             """)
parser.add_argument("--num_iterations", default=60000, type=int)

parser.add_argument("--lr", default=1e-3, type=float)

parser.add_argument("--adapt_lr", default=0.01, type=float)
parser.add_argument("--train_adapt_steps", default=5, type=int)
parser.add_argument("--test_adapt_steps", default=5, type=int)

parser.add_argument("--meta_wrapping", default='maml', type=str)
parser.add_argument("--first_order", default=True, type=bool)

# parser.add_argument("--kd_T", default=2, type=float)
# parser.add_argument("--kd_mode", default='dual', type=str,
#                     choices=['inner', 'outer', 'dual'])
                                            
# parser.add_argument("--lambda_inner_ce", default=1.0, type=float)
# parser.add_argument("--lambda_inner_kd", default=1.0, type=float)
# parser.add_argument("--lambda_outer_ce", default=1.0, type=float)
# parser.add_argument("--lambda_outer_kd", default=0.2, type=float)

parser.add_argument("--teacher_backbone", default='convnext_small.in12k_ft_in1k_384', type=str,
                    choices=['efficientnet_b1', 'wide_resnet50_2', 'convnext_small.in12k_ft_in1k_384'])

parser.add_argument("--device", default='cuda', type=str)

args = parser.parse_args()

if args.train_shot == 1 :
    parser.add_argument("--train_query", default=1, type=int)
    parser.add_argument("--test_shot", default=1, type=int)
    parser.add_argument("--task_batch_size", default=10, type=int)
    parser.add_argument("--kd_T", default=2, type=float)
    parser.add_argument("--kd_mode", default='dual', type=str)
    parser.add_argument("--lambda_inner_ce", default=1.0, type=float)
    parser.add_argument("--lambda_inner_kd", default=1.0, type=float)
    parser.add_argument("--lambda_outer_ce", default=1.0, type=float)
    parser.add_argument("--lambda_outer_kd", default=0.2, type=float)

elif args.train_shot in [5, 10] :
    parser.add_argument("--train_query", default=args.train_shot, type=int)
    parser.add_argument("--test_shot", default=args.train_shot, type=int)
    parser.add_argument("--task_batch_size", default=5, type=int)
    parser.add_argument("--kd_T", default=8, type=float)
    parser.add_argument("--kd_mode", default='outer', type=str)
    parser.add_argument("--lambda_inner_ce", default=1.0, type=float)
    parser.add_argument("--lambda_inner_kd", default=0.0, type=float)
    parser.add_argument("--lambda_outer_ce", default=1.0, type=float)
    parser.add_argument("--lambda_outer_kd", default=1.0, type=float)

args = parser.parse_args()

config = {
        'argparse' : args,
        'save_name_tag' : f'strong_CXs_T{args.kd_T}', ################################################
        'memo' : """lr_scheduler 사용 안 함. 20번마다 validation."""
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


save_name = f"{dname}_{args.way}w{args.train_shot}s_{config['save_name_tag']}_{args.kd_mode}KD" # _in{int(args.lambda_inner_ce*10)}{int(args.lambda_inner_kd*10)}_out{int(args.lambda_outer_ce*10)}{int(args.lambda_outer_kd*10)}

import wandb
run = wandb.init(project="TRAIN_OC")
wandb.run.name = save_name
wandb.run.save()
wandb.config.update(config)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, adaptation_indices, evaluation_indices, 
               teacher_backbone, teacher_learner, student_learner,
               criterion, adaptation_steps, mode, device):
    
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # teacher's data
    feature = teacher_backbone.forward_features(data)
    adaptation_feature = feature[adaptation_indices]
    evaluation_feature = feature[evaluation_indices]

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

    teacher_eval_logit = teacher_learner(evaluation_feature)
    teacher_eval_error = criterion(teacher_eval_logit, evaluation_labels)
    teacher_eval_accuracy = accuracy(teacher_eval_logit, evaluation_labels)

    student_eval_logit = student_learner(evaluation_data)
    student_eval_error = criterion(student_eval_logit, evaluation_labels)

    if mode == 'train' and args.kd_mode in ['outer', 'dual']:
        kd_loss = knowledge_distillation_loss(student_eval_logit, teacher_eval_logit, args.kd_T)
        student_eval_error = args.lambda_outer_ce*student_eval_error + args.lambda_outer_kd*kd_loss
    student_eval_accuracy = accuracy(student_eval_logit, evaluation_labels)

    return teacher_eval_error, teacher_eval_accuracy , student_eval_error, student_eval_accuracy
    

################################## Train ##################################


now = datetime.now()
print(f"Start time: {now.strftime('%Y-%m-%d %H:%M:%S')}")

def meta_train(args):
    seed_fixer(args.seed)

    train_data_transforms = A.Compose([
        # Don't use the Resize and CenterCrop when the dataset is CIFAR_FS.
        # A.Resize(96,96),
        # A.CenterCrop(84,84),

        A.OneOf([
                A.GaussNoise(var_limit=(50.0, 500.0), p=1),
                A.OneOf([A.MultiplicativeNoise(multiplier=(0.6,1.4), p=1), 
                        A.MultiplicativeNoise(multiplier=(0.3,1.7), p=1), ], p=1),# Shot noise
                A.OneOf([A.PixelDropout(dropout_prob=0.05, per_channel=True, p=1), 
                        A.PixelDropout(dropout_prob=0.20, per_channel=True, p=1), ], p=1),# Impulse noise
                A.Defocus(radius=(3, 13), alias_blur=(0.2, 0.7), p=1),
                A.OneOf([A.GlassBlur(sigma=1.0, max_delta=1, p=1),
                        A.GlassBlur(sigma=2.0, max_delta=3, p=1),], p=1),
        
                A.MotionBlur(blur_limit=(7,33), p=1),
                A.ZoomBlur(max_factor=1.5, p=1),
                A.OneOf([A.RandomRain(slant_lower=-7, slant_upper=7, drop_length=7, drop_width=2, drop_color=(255,255,255), 
                                    blur_value=2, brightness_coefficient=1.0, rain_type='drizzle', p=1),
                        A.Compose([A.RandomBrightness(limit=(0.1, 0.15), p=1),
                                    A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=1, drop_width=3, drop_color=(255,255,255), 
                                                blur_value=5, brightness_coefficient=1.0, rain_type='heavy', p=1)])], p=1), # Snow. RandomSnow() X
                # (Frost)
                A.RandomFog(p=1),

                A.RandomBrightness(limit=(0.1, 0.4), p=1),
                A.RandomContrast(limit=(-0.8, -0.2), p=1),
                A.GlassBlur(sigma=1.0, max_delta=5, iterations=3, p=1), # Elastic
                A.OneOf([A.Superpixels(p_replace=0.01, n_segments=150, p=1),
                        A.Superpixels(p_replace=0.01, n_segments=150, max_size=60, p=1),], p=1), # Pixelate
                A.JpegCompression(quality_lower=10, quality_upper=50, p=1),  
            ], p=1),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
        ToTensorV2()
    ])

    test_data_transforms = A.Compose([
        # Already resized in the process of creating the C dataset.
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
        ToTensorV2()
    ])

    print('Making Train Tasksets...')
    trian = make_df(root=args.train_data_dir, mode='train')
    train_dataset = CustomDataset(trian['img_path'].values, trian['label'].values, train_data_transforms)
    train_tasksets = Meta_Transforms(dataset = train_dataset, 
                                    way = args.way, 
                                    shot = args.train_shot, 
                                    query = args.train_query, 
                                    num_tasks = -1)
    
    print('Making Valid Tasksets...')
    valid = make_df(root=args.test_data_dir, mode='validation')
    valid_dataset = CustomDataset(valid['img_path'].values, valid['label'].values, test_data_transforms)
    valid_tasksets = Meta_Transforms(dataset = valid_dataset, 
                                    way = args.way, 
                                    shot = args.test_shot, 
                                    query = args.test_query, 
                                    num_tasks = -1)
    

    train_adapt_idx, train_eval_idx = index_preprocessing(way=args.way, shot=args.train_shot, query=args.train_query)
    test_adapt_idx, test_eval_idx = index_preprocessing(way=args.way, shot=args.test_shot, query=args.test_query)

    if args.dataset_name in ['mini_imagenet', 'CUB_200_2011']:
        img_sige = 84
        hidden_dim = 64
        spatial_size = 5
    elif args.dataset_name == 'CIFAR_FS':
        img_sige = 32
        hidden_dim = 32
        spatial_size = 2

    # maml Model
    teacher_backbone = timm.create_model(args.teacher_backbone, features_only=False, pretrained=True, num_classes=args.way)
    a = teacher_backbone.forward_features(torch.randn(1,3,img_sige,img_sige))
    teacher_backbone.to(device)
    teacher_backbone.eval()

    teacher = CNN1(channel_size=a.size(1), kernel_size=a.size(2), num_classes=args.way)
    teacher_maml = l2l.algorithms.MAML(teacher, lr=args.adapt_lr, first_order=False)
    teacher_maml.to(device)

    student = CNN4(hidden_dim=hidden_dim, spatial_size=spatial_size, num_classes=args.way)
    student_maml = l2l.algorithms.MAML(student, lr=args.adapt_lr, first_order=args.first_order)
    student_maml.to(device)


    criterion = nn.CrossEntropyLoss(reduction='mean')

    t_optimizer = optim.Adam(teacher_maml.parameters(), args.lr)
    s_optimizer = optim.Adam(student_maml.parameters(), args.lr)

    now = datetime.now()
    date = now.strftime('%m%d')
    best_teacher_accuracy = 0
    best_student_accuracy = 0
    for iteration in tqdm(range(1, args.num_iterations+1)):
        train_teacher_accuracy_sum = 0.0
        train_teacher_loss_sum = 0.0
        train_student_accuracy_sum = 0.0
        train_student_loss_sum = 0.0
        
        valid_teacher_accuracy_sum = 0.0
        valid_teacher_loss_sum = 0.0
        valid_student_accuracy_sum = 0.0
        valid_student_loss_sum = 0.0

        t_optimizer.zero_grad()
        s_optimizer.zero_grad()
        for task in range(args.task_batch_size):
            # Train
            teacher_learner = teacher_maml.clone()
            student_learner = student_maml.clone()
            train_batch = train_tasksets.sample()
            teacher_loss, teacher_accuracy, student_loss, student_accuracy = fast_adapt(train_batch,
                                                                                        train_adapt_idx, 
                                                                                        train_eval_idx,
                                                                                        teacher_backbone,
                                                                                        teacher_learner,
                                                                                        student_learner,
                                                                                        criterion,
                                                                                        args.train_adapt_steps,
                                                                                        mode = 'train',
                                                                                        device = device,
                                                                                        )
            train_teacher_accuracy_sum += teacher_accuracy.item()
            train_teacher_loss_sum += teacher_loss.item()
            train_student_accuracy_sum += student_accuracy.item()
            train_student_loss_sum += student_loss.item()
            
            teacher_loss.backward(retain_graph=True) # retain_graph=True
            student_loss.backward()
         
            # Valid
            if iteration==1 or iteration%20==0: # This "if" is optional.
                teacher_learner = teacher_maml.clone()
                student_learner = student_maml.clone()
                valid_batch = valid_tasksets.sample()
                teacher_loss, teacher_accuracy, student_loss, student_accuracy = fast_adapt(valid_batch,
                                                                                            test_adapt_idx, 
                                                                                            test_eval_idx,
                                                                                            teacher_backbone,
                                                                                            teacher_learner,
                                                                                            student_learner,
                                                                                            criterion,
                                                                                            args.test_adapt_steps,
                                                                                            mode = 'valid',
                                                                                            device = device,
                                                                                            )
                valid_teacher_accuracy_sum += teacher_accuracy.item()
                valid_teacher_loss_sum += teacher_loss.item()
                valid_student_accuracy_sum += student_accuracy.item()
                valid_student_loss_sum += student_loss.item() 

        # Average the accumulated gradients and optimize
        for p in teacher_maml.parameters():
            p.grad.data.mul_(1.0 / args.task_batch_size)
        t_optimizer.step()

        for p in student_maml.parameters():
            p.grad.data.mul_(1.0 / args.task_batch_size)
        s_optimizer.step()


        if iteration==1 or iteration%20==0:
            train_teacher_accuracy = train_teacher_accuracy_sum /args.task_batch_size *100
            train_teacher_loss = train_teacher_loss_sum /args.task_batch_size
            train_student_accuracy = train_student_accuracy_sum /args.task_batch_size *100
            train_student_loss = train_student_loss_sum /args.task_batch_size

            valid_teacher_accuracy = valid_teacher_accuracy_sum /args.task_batch_size *100
            valid_teacher_loss = valid_teacher_loss_sum /args.task_batch_size
            valid_student_accuracy = valid_student_accuracy_sum /args.task_batch_size *100
            valid_student_loss = valid_student_loss_sum /args.task_batch_size


            wandb.log({ "Train Teacher Accuracy": train_teacher_accuracy,
                        "Train Teacher loss": train_teacher_loss,
                        "Train Student Accuracy": train_student_accuracy,
                        "Train Student loss": train_student_loss,

                        "Valid Teacher Accuracy": valid_teacher_accuracy,
                        "Valid Teacher loss": valid_teacher_loss,
                        "Valid Student Accuracy": valid_student_accuracy,
                        "Valid Student loss": valid_student_loss,
                        }, step=iteration)
            
            # Model Save
            if iteration > 1 and valid_teacher_accuracy > best_teacher_accuracy : 
                best_teacher_accuracy = valid_teacher_accuracy
                t_name = f"{date}_{save_name}_Teacher.pth"
                t_filepath = os.path.join(args.save_dir, t_name)
                torch.save(teacher_maml.state_dict(), t_filepath)
                print(f"☑️ {iteration}: Best Teacher Model Saved. Valid Acc:{best_teacher_accuracy:.3f}%")

            if iteration > 1 and valid_student_accuracy > best_student_accuracy:
                best_student_accuracy = valid_student_accuracy
                s_name = f"{date}_{save_name}_Student.pth"
                s_filepath = os.path.join(args.save_dir, s_name)
                torch.save(student_maml.state_dict(), s_filepath)
                print(f"✅ {iteration}: Best Student Model Saved. Valid Acc:{best_student_accuracy:.3f}%")
        

        if iteration%5000==0:
            now = datetime.now()
            print(f"{iteration}: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                

if __name__ == '__main__':
    meta_train(args)

now = datetime.now()
print(f"End time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
