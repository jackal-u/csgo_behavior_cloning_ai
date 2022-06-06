import re
import random
import time
from multiprocessing import Pool
from multiprocessing.dummy import freeze_support
import cv2
from network import Net
import argparse
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp

tfms = transforms.Compose([transforms.ToTensor(), transforms.Resize(128), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) # , transforms.ToTensor(),
# https://mapengsen.blog.csdn.net/article/details/117960730?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-117960730-blog-104272600.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-117960730-blog-104272600.pc_relevant_default&utm_relevant_index=1
# 这里的参数是从imageNet中平均出来的，其实并不符合CSGO的实际环境。实际输出非常难看。


def get_frame_from_mp4(path, skip_frames_num=383, show_frames_num = False): # , size=(200, 150)
    """
    读取图像并做图像变换
    :param path:
    :param skip_frames_num:
    :return:
    """

    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    frames = []
    # Preprocess image

    # torch.Size([3, 224, 298])
    while success:
        success, image = vidcap.read()
        if image is None:
            continue
        # print(image, type(image))
        #pi = Image.fromarray(image)
        # print("before transform")
        # image = tfms(image)
        # print("done transform")
        # image = cv2.resize(image, size)
        frames.append(image)
        #print(image.shape)
    if show_frames_num:
        print(len(frames))
    frames = frames[skip_frames_num-1:]
    return frames


def produce_older_steps(frames, time_steps):
    """
    :param frames: a list of single frames shaped CxWxH
    :return: a list of time series frames shaped TxCxWxH
    """
    frames_with_old = []
    for i, frame in enumerate(frames):
        frame_with_old = frames[:i+1][-time_steps:]
        # if no enough frames before, we padding it with zero frames
        if len(frame_with_old) < time_steps:
            frame_with_old = [torch.zeros(frame.size()) for e in range(time_steps-len(frame_with_old))] + frame_with_old
        frame_with_old = torch.stack(frame_with_old)
        frames_with_old.append(frame_with_old)
    return frames_with_old


def exec_load_job(li):
    mp4_path, label_path = li
    frames = get_frame_from_mp4(mp4_path)
    filter_frames = []
    # there are 32fps videos, we need 16 frames per sec only, filter them
    for e in range(len(frames)):
        if e % 2 == 0 and frames[e] is not None:
            frame = tfms(frames[e])
            # frame = torch.from_numpy(frame)
            frame = torch.permute(frame, (0, 2, 1))
            filter_frames.append(frame)
    if not filter_frames:
        print("null ", mp4_path)
        return
    print("shape of each frame ", filter_frames[-1].shape, len(filter_frames))
    frames = filter_frames
    # label_path = names_list[idx].split(' ')[1]
    labels = get_label_by_name(label_path)
    sample = {'frames': frames, 'labels': labels, "frame_path": mp4_path, "label_path": label_path}
    ## process to cuda
    # 1sec = 32frames = 16 filtered frames =  16acitons; thus, filter abnormal data
    frames, labels = sample['frames'], sample['labels']
    detect_bad = len(sample['frames']) / len(sample['labels'])
    if detect_bad > 1.1 or detect_bad < 0.97:
        print(detect_bad, sample["frame_path"], "is bad! skipping it")
        return
    # cut if there are different numbers of filtered_frames&labels
    min_num = min(len(frames), len(labels))
    frames = frames[:min_num]
    # to shape: batch steps channel width height
    frames = produce_older_steps(frames, 16)
    frames = torch.stack(frames)
    inputs = frames  # shape : batch, channel, width, height
    labels = labels[:min_num]
    labels_tensor = torch.stack([each[1] for each in labels])
    print("frames shape", frames.shape)
    return inputs, labels_tensor


def get_label_by_name(path):
    """
    返回 元素为[tick , tensor([        ])]的list
    :param path:
    :return:
    """
    li = []
    path = path.strip("\n")
    with open(path, "r") as f:
        header = f.readline()
        for line in f.readlines():
            acts = line.split("\t")[1:]
            acts = [int(each) for each in acts]
            tick = acts[0]
            none_aim_labels = acts[1:5] + acts[7:]
            li.append([tick, torch.tensor(none_aim_labels + [acts[5], acts[6]])])
        return li


class MyDataset(Dataset):
    def __init__(self, root_dir, names_file, transform=None):
        self.root_dir = root_dir  # 数据集的根目录
        self.names_file = names_file  # meta.csv 的路径
        self.transform = transform  # 预处理函数
        self.size = 0           # 视频个数
        self.names_list = []  # 用来存放meta.csv中的文件名的
        self.data_ram = []
        self.out_ram = []
        # reading from meta
        if not os.path.isfile(self.names_file):
            print("meta.csv NOT FOUND")
            print(self.names_file + 'does not exist!')
        print("meta.csv FOUND")
        file = open(self.names_file)
        for line in file:
            #print("reading lines")
            self.names_list.append(line)
            self.size+=1
        # preload data in ram
        mp4_path = [self.names_list[idx].split(' ')[0] for idx in range(self.size)]
        label_path = [self.names_list[idx].split(' ')[1] for idx in range(self.size)]
        multiple_results = map(exec_load_job, [i for i in zip(mp4_path, label_path)])   #[pool.apply_async(func=exec_load_job, args = (i,)) for i in zip(mp4_path, label_path)]  # 异步开启4个进程（同时而不是依次等待）
        # outs = [res for res in multiple_results]  # 接收4个进程的输出结果 (inputs, labels_tensor)
        # data_ram content = (video_frames, labels), each video_frames has 406 frames, each frame have 16 subframes(previous)
        self.data_ram = [(each[0].half(), each[1]) for each in multiple_results if each is not None]
        # change data_ram content to = (previous_frames, label)
        for each_video in self.data_ram:
            frames, labels = each_video  # frames = [subframes_1, sub_frames_2, ..., sub_frames_406]
            for frame, label in zip(frames, labels):
                self.out_ram.append((frame, label))
        self.data_ram = []
        self.size = len(self.out_ram)
        print("load complete")

    def __len__(self):
        #print("the length means the num of videos")
        return self.size

    def __getitem__(self, idx):
        return self.out_ram[idx]

    def shuffle(self):
        random.shuffle(self.out_ram)


if __name__ == '__main__':

                parser = argparse.ArgumentParser(description='Process some integers.')
                parser.add_argument('-lr', required=True, type=float,
                                    help='learning rate')
                parser.add_argument('-name', required=True, type=str,
                                    help='model name, e.g: ./net_20.pkl')
                parser.add_argument('-epoch', default=200, type=int,
                                    help='epoch num')
                parser.add_argument('-batch', default=50, type=int,
                                    help='batch num, default 50')
                parser.add_argument("--local_rank", type=int)
                args = parser.parse_args()
                model_name = args.name
                lr = args.lr
                batch_size = args.batch
                epoch_num = args.epoch
                model_train_num = re.search(r"\d+", model_name)[0]
                writer = SummaryWriter(r"/root/tf-logs/lr_{}_efnetb0_pretrained".format(lr))

                try:
                    state_dict = torch.load(model_name)
                    have_previous = True
                    net = Net(have_previous).half()  # 16位精度训练
                    net.load_state_dict(state_dict)
                    print("load previous ", model_name)
                except:
                    import traceback
                    traceback.print_exc()
                    have_previous = False
                    print("no such net work found, starting by net_0.pkl")
                    net = Net(have_previous).half()
                    pass
                net.train()
                # torch.cuda.set_device(args.local_rank)
                # torch.distributed.init_process_group(backend="nccl")
                # net = torch.nn.parallel.DistributedDataParallel(net.cuda())  # device_ids will include all GPU devices by default
                net = torch.nn.DataParallel(net.cuda())  #, device_ids=device_ids, output_device=device_ids[0]
                # calculate params num
                num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
                model_size = num_params/1000000
                print(f'The number of parameters of model is{num_params}, size: {model_size} mb', )

                # define losses and optimizer
                import torch.optim as optim

                criterion = nn.CrossEntropyLoss().cuda()
                optimizer = optim.AdamW(net.parameters(),  eps=0.0001) #  lr=lr,   # for nan, see : https://discuss.pytorch.org/t/adam-half-precision-nans/1765/9
                dataset = MyDataset(root_dir="./", names_file="./meta.csv", transform=None)
                data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, sampler=SequentialSampler(dataset),
                                         num_workers=0, pin_memory=False
                                         )

                for epoch in range(epoch_num):
                    running_loss = 0.0
                    vid_n = 0
                    # dataset.shuffle()
                    i=0
                    for inputs,labels_tensor in data_loader:
                        t0 = time.time()
                        # print("STARTING {} round of training".format(i))
                        inputs, labels_tensor = inputs.cuda(), labels_tensor.cuda()
                        #print("load time cost", time.time()-t0)
                        vid_n = vid_n + 1
                        # frames, labels = sample['frames'], sample['labels']
                        # detect_bad = len(sample['frames'])/len(sample['labels'])
                        # if detect_bad > 1.1 or detect_bad < 1:
                        #     print(detect_bad, sample["frame_path"], "is bad! skipping it")
                        #     continue
                        # vid_n = vid_n + 1
                        # min_num = min(len(frames), len(labels))
                        # frames = frames[:min_num]
                        # frames = torch.stack(frames)
                        # inputs = frames  # shape : batch, channel, width, height
                        # labels = labels[:min_num]
                        # labels_tensor = torch.stack([each[1] for each in labels])
                        # before_put_image = torch.cuda.memory_allocated()
                        # # 图片大小计算方法： 800*600*3*4*个数/1000000000 = XGB
                        # inputs = inputs.half().cuda()
                        # labels_tensor = labels_tensor.cuda()
                        t1 = time.time()
                        optimizer.zero_grad(set_to_none=True)
                        # before_forward = torch.cuda.memory_allocated()

                        output = net(inputs)
                        #print("forward took time ", time.time() - t1)
                        # t2 = time.time()
                        w,a,s,d,fire,scope,jump,crouch,walking,reload,e,switch,aim_x,aim_y = output
                        l_w, l_a, l_s, l_d, l_fire, l_scope, l_jump = labels_tensor[:,[0]].squeeze(),labels_tensor[:,[1]].squeeze(),labels_tensor[:,[2]].squeeze(),labels_tensor[:,[3]].squeeze(),labels_tensor[:,[4]].squeeze(),labels_tensor[:,[5]].squeeze(),labels_tensor[:,[6]].squeeze()
                        l_crouch, l_walking, l_reload, l_e, l_switch, l_aim_x, l_aim_y = labels_tensor[:,[7]].squeeze(),labels_tensor[:,[8]].squeeze(),labels_tensor[:,[9]].squeeze(),labels_tensor[:,[10]].squeeze(),labels_tensor[:,[11]].squeeze(),labels_tensor[:,[12]].squeeze(),labels_tensor[:,[13]].squeeze()
                        # define loss
                        loss_w_v, loss_a_v, loss_s_v, loss_d_v = criterion(w,l_w),criterion(a,l_a),criterion(s,l_s),criterion(d,l_d)
                        #wprint("aim_x ",aim_x[0], "aimx_label", l_aim_x[0] )
                        #print("w", w[0], "l_w", l_w[0])
                        loss_fire_v, loss_scope_v, loss_jump_v, loss_crouch_v, loss_walking_v, loss_reload_v, loss_e_v, loss_switch_v = criterion(fire,l_fire),criterion(scope,l_scope),criterion(jump,l_jump),criterion(crouch,l_crouch),criterion(walking,l_walking),criterion(reload,l_reload),criterion(e,l_e),criterion(switch,l_switch)
                        loss_aim_x_v, loss_aim_y_v = criterion(aim_x, l_aim_x), criterion(aim_y,l_aim_y)
                        #  print("aim_y.shape, l_aim_y.shape ", aim_y.shape, l_aim_y.shape)
                        #  torch.Size([143, 33]) torchSize([143])
                        loss = loss_w_v + loss_a_v + loss_s_v + loss_d_v + 50*loss_fire_v + loss_scope_v + loss_jump_v + loss_crouch_v + loss_walking_v + loss_reload_v + loss_e_v + loss_switch_v + 10*loss_aim_x_v + 10*loss_aim_y_v
                        loss_list = [loss_w_v.item(),loss_a_v.item(), loss_s_v.item(), loss_d_v.item(), loss_fire_v.item(), loss_scope_v.item(), loss_jump_v.item(), loss_crouch_v.item(), loss_walking_v.item(), loss_reload_v.item(), loss_e_v.item(), loss_switch_v.item(), loss_aim_x_v.item(), loss_aim_y_v.item()]
                        loss_name = ['loss_w','loss_a','loss_s','loss_d','loss_fir','loss_scop','loss_jump','loss_crouch','loss_walking','loss_reload','loss_e','loss_switch','loss_aim_x','loss_aim_y']
                        writer.add_scalars('Loss/each', dict(zip(loss_name, loss_list)), epoch*len(dataset)+i)
                        loss_value = sum(loss_list)
                        writer.add_scalar('Loss/total', loss_value, epoch*len(dataset)+i)

                        #print("Loss time cost ", time.time()-t2)
                        t3 = time.time()
                        loss.backward()
                        optimizer.step()
                        running_loss += loss_value

                        # calculate correct rate for each action
                        # w_acc = (torch.argmax(w, dim=1)==l_w).sum()/l_w.shape[0]
                        fire_acc = (torch.argmax(fire, dim=1) == l_fire).sum() / l_fire.shape[0]
                        # # calculate similarity with the batch average
                        # w_sim = torch.argmax(w, dim=1).sum()/l_w.shape[0]
                        # fire_sim = torch.argmax(fire, dim=1).sum() / l_fire.shape[0]
                        # writer.add_scalar('accuracy/w_acc', w_acc.item(), epoch * len(dataset) + i)
                        writer.add_scalar('accuracy/fire_acc', fire_acc.item(), epoch * len(dataset) + i)
                        # writer.add_scalar('sim/w_sim', w_sim.item(), epoch * len(dataset) + i)
                        # writer.add_scalar('sim/fire_sim', fire_sim.item(), epoch * len(dataset) + i)
                        i += 1
                        # print("backward took time ", time.time() - t3, "  total time ", time.time() - t0)
                    writer.flush()

                    print(f'{epoch + 1},   loss: {running_loss / vid_n:.3f}', vid_n)

                    if epoch%20==0:
                        torch.save(net.module.state_dict(), "net_{}.pkl".format(str(int(epoch))))
                        print("saving dataparallel model {}".format("net_{}.pkl".format(epoch)))


