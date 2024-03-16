from __future__ import print_function
import torch
print(torch.__version__)
import torch.utils.data
import torchvision.transforms as transforms
from basic_code import data_generator

cate2label = {
                'CK+':{0: 'Happy', 1: 'Angry', 2: 'Disgust', 3: 'Fear', 4: 'Sad', 6: 'Contempt', 5: 'Surprise',
                     'Angry': 1,'Disgust': 2,'Fear': 3,'Happy': 0,'Contempt': 6,'Sad': 4,'Surprise': 5},
                # 'CK+':{0: 'Happy', 1: 'Angry', 2: 'Disgust', 3: 'Fear', 4: 'Sad', 5: 'Contempt', 6: 'Surprise',
                #      'Angry': 1,'Disgust': 2,'Fear': 3,'Happy': 0,'Contempt': 5,'Sad': 4,'Surprise': 6},
              'AFEW':{0: 'Happy',1: 'Angry',2: 'Disgust',3: 'Fear',4: 'Sad',5: 'Neutral',6: 'Surprise',
                  'Angry': 1,'Disgust': 2,'Fear': 3,'Happy': 0,'Neutral': 5,'Sad': 4,'Surprise': 6},
                #   for compare ck+
                'RAVDESS':{0: 'happy',1: 'angry',2: 'disgust',3: 'fearful',4: 'sad',5: 'surprised',
                    'happy': 0,'angry': 1,'disgust': 2,'fearful': 3,'sad': 4,'surprised': 5},
              #  for use ravdess model
            #   'RAVDESS':{0: 'neutral',1: 'calm',2: 'happy',3: 'sad',4: 'angry',5: 'fearful',6: 'disgust',7: 'surprised',
            #       'neutral': 0,'calm': 1,'happy': 2,'sad': 3,'angry': 4,'fearful': 5,'disgust': 6,'surprised': 7},
            # for compare ck+
              'OULU':{0: 'Happy', 1: 'Angry', 2: 'Disgust', 3: 'Fear', 4: 'Sadness', 5: 'Surprise',
                     'Happy': 0, 'Angry': 1,'Disgust': 2,'Fear': 3,'Sadness': 4,'Surprise': 5},
            #  for use oulu model
            #    'OULU':{0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sadness', 5: 'Surprise',
            #           'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,'Sadness': 4,'Surprise': 5}
                      }

def ckplus_faces_baseline(video_root, video_list, fold, batchsize_train, batchsize_eval):
    train_dataset = data_generator.TenFold_VideoDataset(
                                        video_root=video_root,
                                        video_list=video_list,
                                        rectify_label=cate2label['CK+'],
                                        transform=transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
                                        fold=fold,
                                        run_type='train'
                                        )

    val_dataset = data_generator.TenFold_VideoDataset(
                                        video_root=video_root,
                                        video_list=video_list,
                                        rectify_label=cate2label['CK+'],
                                        transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
                                        fold=fold,
                                        run_type='test'
                                        )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize_train, shuffle=True, num_workers=8,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batchsize_eval, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, val_loader

def ckplus_faces_fan(video_root, video_list, fold, batchsize_train, batchsize_eval):
    train_dataset = data_generator.TenFold_TripleImageDataset(
                                        video_root=video_root,
                                        video_list=video_list,
                                        rectify_label=cate2label['CK+'],
                                        transform=transforms.Compose([
                                            transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
                                        fold=fold,
                                        run_type='train',
                                        )

    val_dataset = data_generator.TenFold_VideoDataset(
                                        video_root=video_root,
                                        video_list=video_list,
                                        rectify_label=cate2label['CK+'],
                                        transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
                                        fold=fold,
                                        run_type='test'
                                        )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize_train, shuffle=True, num_workers=8,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batchsize_eval, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, val_loader

def afew_faces_baseline(root_train, list_train, batchsize_train, root_eval, list_eval, batchsize_eval):

    train_dataset = data_generator.VideoDataset(
        video_root=root_train,
        video_list=list_train,
        rectify_label=cate2label['AFEW'],
        transform=transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
    )

    val_dataset = data_generator.VideoDataset(
        video_root=root_eval,
        video_list=list_eval,
        rectify_label=cate2label['AFEW'],
        transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
        csv=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize_train, shuffle=True,
        num_workers=8, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize_eval, shuffle=False,
        num_workers=8, pin_memory=True)
    return train_loader, val_loader

def afew_faces_fan(root_train, list_train, batchsize_train, root_eval, list_eval, batchsize_eval):

    train_dataset = data_generator.TripleImageDataset(
        video_root=root_train,
        video_list=list_train,
        rectify_label=cate2label['AFEW'],
        transform=transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
    )

    val_dataset = data_generator.VideoDataset(
        video_root=root_eval,
        video_list=list_eval,
        rectify_label=cate2label['AFEW'],
        transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
        csv=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize_train, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize_eval, shuffle=False,
        num_workers=8, pin_memory=True)


    return train_loader, val_loader
def rav_faces_baseline(root_train, list_train, batchsize_train, root_eval, list_eval, batchsize_eval):

    train_dataset = data_generator.VideoDataset(
        video_root=root_train,
        video_list=list_train,
        rectify_label=cate2label['RAVDESS'],
        transform=transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
    )

    val_dataset = data_generator.VideoDataset(
        video_root=root_eval,
        video_list=list_eval,
        rectify_label=cate2label['RAVDESS'],
        transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
        csv=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize_train, shuffle=True,
        num_workers=8, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize_eval, shuffle=False,
        num_workers=8, pin_memory=True)
    return train_loader, val_loader

def rav_faces_fan(root_train, list_train, batchsize_train, root_eval, list_eval, batchsize_eval):

    train_dataset = data_generator.TripleImageDataset(
        video_root=root_train,
        video_list=list_train,
        rectify_label=cate2label['RAVDESS'],
        transform=transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
    )

    val_dataset = data_generator.VideoDataset(
        video_root=root_eval,
        video_list=list_eval,
        rectify_label=cate2label['RAVDESS'],
        transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
        csv=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize_train, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize_eval, shuffle=False,
        num_workers=8, pin_memory=True)


    return train_loader, val_loader
def oulu_faces_fan(root_train, list_train, batchsize_train, root_eval, list_eval, batchsize_eval):

    train_dataset = data_generator.TripleImageDataset(
        video_root=root_train,
        video_list=list_train,
        rectify_label=cate2label['OULU'],
        transform=transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
    )

    val_dataset = data_generator.VideoDataset(
        video_root=root_eval,
        video_list=list_eval,
        rectify_label=cate2label['OULU'],
        transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
        csv=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize_train, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize_eval, shuffle=False,
        num_workers=8, pin_memory=True)
    return train_loader, val_loader
def oulu_faces_fan_ck(root_train, list_train, batchsize_train, root_eval, list_eval, batchsize_eval):

    train_dataset = data_generator.VideoDataset(
        video_root=root_train,
        video_list=list_train,
        rectify_label=cate2label['CK+'],
        transform=transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
    )

    val_dataset = data_generator.VideoDataset(
        video_root=root_eval,
        video_list=list_eval,
        rectify_label=cate2label['OULU'],
        transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
        csv=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize_train, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize_eval, shuffle=False,
        num_workers=8, pin_memory=True)
    return train_loader, val_loader
def model_parameters(_structure, _parameterDir):

    checkpoint = torch.load(_parameterDir)
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = _structure.state_dict()

    for key in pretrained_state_dict:
        if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):

            pass
        else:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]

    _structure.load_state_dict(model_state_dict)
    model = torch.nn.DataParallel(_structure).cuda()

    return model
