import os
from datetime import datetime
import torch
from torch import nn, optim
import learn2learn as l2l
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import timm
from models.base_model import CNN4, CNN1
from utils.util import seed_fixer, index_preprocessing, knowledge_distillation_loss, soft_CrossEntropy_loss, MetaWeights
from utils.CustomDatasets import make_df, CustomDataset, Meta_Transforms

import warnings
warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=2023, type=int)
parser.add_argument("--dataset_name", default='mini_imagenet', type=str,
                    choices=['MiniImagenet', 'mini_imagenet', 'mini-imagenet',
                             'cub', 'CUB', 'CUB_200_2011',
                            'TieredImagenet', 'tiered_imagenet', 'tiered-imagenet',
                            'FC100', 'fc100', 'CIFARFS', 'cifarfs'])
# parser.add_argument("--aug_mode", default='albumentations', type=str,
#                     choices=['torchvision', 'transforms',
#                              'albumentations', 'Albumentations', 'alb', 'Alb'])

parser.add_argument("--original_data_dir", default='/home/hjb3880/WORKPLACE/datasets/mini_imagenet', type=str)
parser.add_argument("--corruption_data_dir", default='/home/hjb3880/WORKPLACE/datasets/mini_imagenet_C', type=str)
parser.add_argument("--save_dir", default='saved_models_important2', type=str)

parser.add_argument("--way", default=5, type=int)
parser.add_argument("--train_shot", default=1, type=int)
parser.add_argument("--train_query", default=1, type=int)
parser.add_argument("--test_shot", default=1, type=int)
parser.add_argument("--test_query", default=15, type=int)

parser.add_argument("--task_batch_size", default=12, type=int,
                    help="Recommand: Use 12 when 1-shot and use 6 when 5&10-shot.")
parser.add_argument("--num_iterations", default=60000, type=int)

parser.add_argument("--lr", default=1e-3, type=float)
# parser.add_argument("--scheduler_step", default=30, type=int)
parser.add_argument("--scheduler_gamma", default=0.5, type=float)

parser.add_argument("--adapt_lr", default=0.01, type=float)
parser.add_argument("--train_adapt_steps", default=5, type=int)
parser.add_argument("--test_adapt_steps", default=5, type=int)

parser.add_argument("--meta_wrapping", default='maml', type=str)
parser.add_argument("--first_order", default=True, type=bool)

parser.add_argument("--kd_T", default=8, type=float)
                                                                  # gpu1       gpu0
parser.add_argument("--lambda_inner_ce", default=1, type=float) # 1  0.5       1    1    1  0.9  0.8
parser.add_argument("--lambda_inner_kd", default=1, type=float) # 1  0.5     0.7  0.5  0.3  0.1  0.2
parser.add_argument("--lambda_outer_ce", default=0.5, type=float)
parser.add_argument("--lambda_outer_kd", default=0.5, type=float)

# out  in |  1,1  |  0.9,0.1 |     0.5,0.5 |  1,0.3 |
# 1,1     |    
# 1,0.3   |        
# 0.9,0.1 |  
# 0.5,0.5 |

parser.add_argument("--teacher_backbone", default='convnext_small.in12k_ft_in1k_384', type=str,
                    choices=['efficientnet_b1', 'mobilenetv3_largel_100', 'wide_resnet50_2', 'convnextv2_base.fcmae_ft_in22k_in1k_384', 'convnext_small.in12k_ft_in1k_384'])


parser.add_argument("--device", default='cuda', type=str)

args = parser.parse_args()

config = {
        'argparse' : args,
        'save_name_tag' : f'strong_CXs_2KD', ################################################
        'readme' : 'valid_tasksets:-1'
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

save_name = f"{dname}_{args.way}w{args.train_shot}s_{config['save_name_tag']}_{args.train_adapt_steps}step_in{int(args.lambda_inner_ce*10)}{int(args.lambda_inner_kd*10)}_out{int(args.lambda_outer_ce*10)}{int(args.lambda_outer_kd*10)}" #

import wandb
run = wandb.init(project="Train_OC")
wandb.run.name = save_name
wandb.run.save()
wandb.config.update(config)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, adaptation_indices, evaluation_indices, 
               teacher_backbone, teacher_learner, student_learner, #weight_learner,
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

        kd_loss = knowledge_distillation_loss(student_adapt_logit, teacher_adapt_logit, args.kd_T)
        student_adapt_error = args.lambda_inner_ce*student_adapt_error + args.lambda_inner_kd*kd_loss

        #student_adapt_error = weight_learner(student_adapt_error, kd_loss)

        student_learner.adapt(student_adapt_error) #, allow_unused=True
        # print('weight_learner:', list(weight_learner.parameters())[-1].grad)
        # print('student_learner:', list(student_learner.parameters())[-1].grad)

    teacher_eval_logit = teacher_learner(evaluation_feature)
    teacher_evaluation_error = criterion(teacher_eval_logit, evaluation_labels)
    teacher_evaluation_accuracy = accuracy(teacher_eval_logit, evaluation_labels)

    student_eval_logit = student_learner(evaluation_data)
    student_evaluation_error = criterion(student_eval_logit, evaluation_labels)
    if mode == 'train':
        kd_loss = knowledge_distillation_loss(student_eval_logit, teacher_eval_logit, args.kd_T)
        student_evaluation_error = args.lambda_outer_ce*student_evaluation_error + args.lambda_outer_kd*kd_loss
    student_evaluation_accuracy = accuracy(student_eval_logit, evaluation_labels)

    return teacher_evaluation_error, teacher_evaluation_accuracy , student_evaluation_error, student_evaluation_accuracy
    


#==========================================================================================================


################################## Train ##################################


now = datetime.now()
print(f"Start time: {now.strftime('%Y-%m-%d %H:%M:%S')}")

def meta_train(args):
    seed_fixer(args.seed)

    train_data_transforms = A.Compose([
        #A.Resize(84,84),

        A.Resize(96,96),
        A.CenterCrop(84,84),

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
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
        ToTensorV2()
    ])


    trian = make_df(root=args.original_data_dir, mode='train')
    train_dataset = CustomDataset(trian['img_path'].values, trian['label'].values, train_data_transforms)
    print('Making Train Tasksets...')
    train_tasksets = Meta_Transforms(dataset = train_dataset, 
                                    way = args.way, 
                                    shot = args.train_shot, 
                                    query = args.train_query, 
                                    num_tasks = -1)
    
    valid = make_df(root=args.corruption_data_dir, mode='validation')
    valid_dataset = CustomDataset(valid['img_path'].values, valid['label'].values, test_data_transforms)
    print('Making Valid Tasksets...')
    valid_tasksets = Meta_Transforms(dataset = valid_dataset, 
                                    way = args.way, 
                                    shot = args.test_shot, 
                                    query = args.test_query, 
                                    num_tasks = -1)
    

    train_adapt_idx, train_eval_idx = index_preprocessing(way=args.way, shot=args.train_shot, query=args.train_query)
    test_adapt_idx, test_eval_idx = index_preprocessing(way=args.way, shot=args.test_shot, query=args.test_query)


    # maml Model
    teacher_backbone = timm.create_model(args.teacher_backbone, features_only=False, pretrained=True, num_classes=args.way)
    a = teacher_backbone.forward_features(torch.randn(1,3,84,84))
    teacher_backbone.to(device)

    teacher = CNN1(channel_size=a.size(1), kernel_size=a.size(2), num_classes=args.way)
    teacher_maml = l2l.algorithms.MAML(teacher, lr=args.adapt_lr, first_order=False)
    teacher_maml.to(device)

    student = CNN4(num_classes=args.way)
    if args.meta_wrapping == 'maml' :
        student_maml = l2l.algorithms.MAML(student, lr=args.adapt_lr, first_order=args.first_order)
    elif args.meta_wrapping == 'metasgd' :
        student_maml = l2l.algorithms.MetaSGD(student, lr=0.01, first_order=args.first_order)
        args.train_adapt_steps = 1
        args.test_adapt_steps = 1
    student_maml.to(device)

    # metaweight = MetaWeights()
    # weight_maml = l2l.algorithms.MAML(metaweight, lr=args.adapt_lr, first_order=False)


    criterion = nn.CrossEntropyLoss(reduction='mean')

    t_optimizer = optim.Adam(teacher_maml.parameters(), args.lr)
    s_optimizer = optim.Adam(student_maml.parameters(), args.lr)
    # t_lr_scheduler = optim.lr_scheduler.StepLR(t_optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    # s_lr_scheduler = optim.lr_scheduler.StepLR(s_optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    t_lr_scheduler = optim.lr_scheduler.MultiStepLR(t_optimizer, milestones=[20000, 35000, 45000, 50000, 55000], gamma=args.scheduler_gamma)
    s_lr_scheduler = optim.lr_scheduler.MultiStepLR(s_optimizer, milestones=[20000, 35000, 45000, 50000, 55000], gamma=args.scheduler_gamma)

    now = datetime.now()
    date = now.strftime('%m%d')
    best_teacher_accuracy = 0
    best_student_accuracy = 0
    for iteration in tqdm(range(1, args.num_iterations+1)):
        train_teacher_loss_sum = 0.0
        train_teacher_accuracy_sum = 0.0
        train_student_loss_sum = 0.0
        train_student_accuracy_sum = 0.0

        valid_teacher_loss_sum = 0.0
        valid_teacher_accuracy_sum = 0.0
        valid_student_loss_sum = 0.0
        valid_student_accuracy_sum = 0.0

        t_optimizer.zero_grad()
        s_optimizer.zero_grad()
        for task in range(args.task_batch_size):
            # Train
            teacher_learner = teacher_maml.clone()
            student_learner = student_maml.clone()
            #weight_learner = weight_maml.clone()
            train_batch = train_tasksets.sample()
            teacher_loss, teacher_accuracy, student_loss, student_accuracy = fast_adapt(train_batch,
                                                                                        train_adapt_idx, 
                                                                                        train_eval_idx,
                                                                                        teacher_backbone,
                                                                                        teacher_learner,
                                                                                        student_learner,
                                                                                        #weight_learner,
                                                                                        criterion,
                                                                                        args.train_adapt_steps,
                                                                                        mode = 'train',
                                                                                        device = device,
                                                                                        )
            train_teacher_loss_sum += teacher_loss.item()
            train_teacher_accuracy_sum += teacher_accuracy.item()
            train_student_loss_sum += student_loss.item()
            train_student_accuracy_sum += student_accuracy.item()

            
            teacher_loss.backward(retain_graph=True) # retain_graph=True
            student_loss.backward()

            # Valid
            teacher_learner = teacher_maml.clone()
            student_learner = student_maml.clone()
            valid_batch = valid_tasksets.sample()
            teacher_loss, teacher_accuracy, student_loss, student_accuracy = fast_adapt(valid_batch,
                                                                                        test_adapt_idx, 
                                                                                        test_eval_idx,
                                                                                        teacher_backbone,
                                                                                        teacher_learner,
                                                                                        student_learner,
                                                                                        #weight_learner,
                                                                                        criterion,
                                                                                        args.test_adapt_steps,
                                                                                        mode = 'valid',
                                                                                        device = device,
                                                                                        )
            valid_teacher_loss_sum += teacher_loss.item()
            valid_teacher_accuracy_sum += teacher_accuracy.item()
            valid_student_loss_sum += student_loss.item() 
            valid_student_accuracy_sum += student_accuracy.item()

            
        # Average the accumulated gradients and optimize
        for p in teacher_maml.parameters():
            p.grad.data.mul_(1.0 / args.task_batch_size)
        t_optimizer.step()
        t_lr_scheduler.step()

        for p in student_maml.parameters():
            p.grad.data.mul_(1.0 / args.task_batch_size)
        s_optimizer.step()
        s_lr_scheduler.step()


        # Print some metrics
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
                    "Valid Teacher Accuracy": valid_teacher_accuracy,
                    "Valid Teacher loss": valid_teacher_loss,

                    "Train Student Accuracy": train_student_accuracy,
                    "Train Student loss": train_student_loss,
                    "Valid Student Accuracy": valid_student_accuracy,
                    "Valid Student loss": valid_student_loss,
                    }, step=iteration)

        # 모델 저장하기
        if iteration > 5 and valid_teacher_accuracy > best_teacher_accuracy : 
            best_teacher_accuracy = valid_teacher_accuracy
            t_name = f"{date}_{save_name}_Teacher.pth"
            t_filepath = os.path.join(args.save_dir, t_name)
            torch.save(teacher_maml.state_dict(), t_filepath)
            print(f"☑️ {iteration}: Best Teacher Model Saved. Valid Acc:{best_teacher_accuracy:.3f}%")

        if iteration > 5 and valid_student_accuracy > best_student_accuracy:
            best_student_accuracy = valid_student_accuracy
            s_name = f"{date}_{save_name}_Student.pth"
            s_filepath = os.path.join(args.save_dir, s_name)
            torch.save(student_maml.state_dict(), s_filepath)
            print(f"✅ {iteration}: Best Student Model Saved. Valid Acc:{best_student_accuracy:.3f}%")

        
                

if __name__ == '__main__':
    meta_train(args)

now = datetime.now()
print(f"End time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
