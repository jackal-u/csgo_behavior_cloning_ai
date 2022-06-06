import torch
import keyboard,time
model_name = r'./model/net_20 (3).pkl'
from queue import Queue

import pyautogui
import win32api
import win32con

from PIL import ImageGrab
import win32gui
from torchvision import transforms
from PIL import Image
from pynput.keyboard import Key, Controller
import time
# #
# handle = api.CSAPI(r"D:\PROJECT\BOT\api\csgo.json")

class KeyControll(object):
    def __init__(self):
        self.keyboard = Controller()
        self.queue = Queue()
        self.last_actions = []

    def press_key(self, all_action_li):
        for each_key in all_action_li:
            self.keyboard.press(each_key)
        self.last_actions = all_action_li

    def release_key(self):
        if not self.last_actions:
            return
        for each_key in  self.last_actions:
            self.keyboard.release(each_key)

class Aim(object):
    def __init__(self):
        window_handle = win32gui.FindWindow(None, u"Counter-Strike: Global Offensive - Direct3D 9")
        win32gui.SetForegroundWindow(window_handle)

    def set_aim(self, aim_list):
        sensitivity = 2.5
        m_pitch = 0.022
        m_yaw = 0.022
        pitch, yaw = aim_list
        #print("aim_list", aim_list)
        pixel_y = pitch / (sensitivity * m_pitch)  # sensitivity = 2.5;m_pitch = 0.022
        pixel_x = yaw / (sensitivity * m_yaw)  # sensitivity = 2.5;m_yaw= 0.022

        # y正的是向下走，这个跟内存中上负下正是一样的，因而取正值
        # x的是往右走是正，这跟内存中是反过来的，取负值
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -int(pixel_x), int(pixel_y), 0, 0)

    def set_attack(self):
        pyautogui.leftClick()

    def set_attack2(self):
        pyautogui.rightClick()


aim = Aim()
key_controll = KeyControll()
# time.sleep(0.124)
# def take_action(li):


tfms = transforms.Compose([transforms.Resize(128), transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
hwnd = win32gui.FindWindow(None, u"Counter-Strike: Global Offensive - Direct3D 9")
frame_pool = [torch.zeros(size = (3,162,128)).half().cuda() for i in range(16)]
soft_max = torch.nn.Softmax()
from network_run import Net
with torch.no_grad():
    # LOAD NET
    state_dict = torch.load(model_name)
    have_previous = True
    net = Net(have_previous).half()  # 16位精度训练
    net.load_state_dict(state_dict)
    net = net.cuda()
    net.eval()
    while True:
        t0 = time.time()
        if keyboard.is_pressed("q"):
            time.sleep(1)
            break
        ## run network
        # catch frame\
        dimensions = win32gui.GetWindowRect(hwnd)
        image = ImageGrab.grab(dimensions)
        # image.show()
        image = tfms(image)
        input = torch.permute(image, (0, 2, 1)).half().cuda()
        frame_pool.append(input)
        frame_pool = frame_pool[-16:]
        input = torch.stack(frame_pool)
        print("input shape", input.shape)
        # forward
        ouput = net(input)
        win32gui.SetForegroundWindow(hwnd)
        ## take action
        w,a,s,d,fire,scope,jump,crouch,walking,reload,e,switch,aim_x,aim_y = ouput
        out = [w,a,s,d,fire,scope,jump,crouch,walking,reload,e,switch,aim_x,aim_y]
        # print("w", w.shape)
        #print("out", out)
        #w,a,s,d,is_fire,is_scope,is_jump,is_crouch,is_walking,is_reload,is_e,switch
        action = [torch.argmax(soft_max(a)).item() for a in out]
        print(action[0], action)

        ## apply action
        li = action
        # out =  [0 ,w,a,s,d,fire,scope,jump,crouch,walking,reload,e,switch, aim_x,aim_y]
        # 0,w,a,s,d,is_fire,is_scope,is_jump,is_crouch,is_walking,is_reload,is_e,switch,aim_x,aim_y
        li = [0] + li
        print("action list", li)
        w, a, s, d = li[1:5]
        # 得到瞄准角度
        mouse_x, mouse_y = li[-2:]
        mouse_y_possibles = [-50.0, -40.0, -30.0, -20.0, -18.0, -16.0, -14.0, -12.0, -10.0, -8.0, -6.0, -5.0, -4.0,
                             -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0,
                             20.0, 30.0, 40.0, 50.0]
        mouse_x_possibles = [-170, -130, -100, -80, -70, -60, -50.0, -40.0, -30.0, -20.0, -18.0, -16.0, -14.0, -12.0,
                             -10.0, -8.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0,
                             10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 30.0, 40.0, 50.0, 60, 70, 80, 100, 130, 170]
        try:
            pix_x, pix_y = int(mouse_x_possibles[mouse_x]), int(mouse_y_possibles[mouse_y])
        except IndexError:
            print(mouse_x, mouse_y)
            exit()
        # 将行为映射成按键
        is_fire, is_scope, is_jump, is_crouch, is_walking, is_reload, is_e, switch = li[5:-2]
        action_li = []
        if w:
            action_li.append("w")
        if a:
            action_li.append("a")
        if s:
            action_li.append("s")
        if d:
            action_li.append("d")
        if is_fire:
            aim.set_attack()
        if is_scope:
            aim.set_attack2()
        if is_jump:
            action_li.append(" ")
        if is_crouch:
            action_li.append(Key.ctrl)
        if is_walking:
            action_li.append(Key.shift)
        if is_reload:
            action_li.append("r")
        if is_e:
            action_li.append("e")
        action_li.append(str(switch))
        # print("action_li", action_li)
        # 瞄准
        print("aim_li", [pix_y, pix_x])
        aim.set_aim([pix_y, pix_x])
        key_controll.release_key()
        #time.sleep(abs(1 / 16 - (time.time() - t0)))
        key_controll.press_key(action_li)
        print("loop time cost: ",  (time.time()-t0))
        # time.sleep(1/16 - (time.time()-t0))




