import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')  # todo:change by image size
#     def forward(self, x):
#         x = self.efficient_net(x)
#         return x
#
# net = Net()
# num_params = sum(p.numel() for p in net2.parameters() if p.requires_grad)
# model_size = num_params / 1000000
# print(f'The number of parameters of model is{num_params}, size: {model_size} mb', )

class Net(nn.Module):
    def __init__(self, load):  #
        super().__init__()
        self.fc1 = nn.Linear(1280, 512)  # todo:change by image size
        self.fc2 = nn.Linear(512, 200)
        self.fc3 = nn.Linear(200, 200)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        # if load:
        #     # if there is a previous model of ours, we use our model and load into it
        #     self.efficient_net = EfficientNet.from_name('efficientnet-b0')
        # else:
        #     self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        self.convlstm = ConvLSTM(input_dim=1280,
                                 hidden_dim=[128, 64, 64],
                                 kernel_size=(3, 3),
                                 num_layers=3,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=True)

        self.fc_aim_x = nn.Sequential(
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 45))
        self.fc_aim_y = nn.Sequential(
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.Linear(200, 33))

        self.fc_w = nn.Sequential(
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 2))

        self.fc_a = nn.Sequential(
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 2))
        self.fc_s = nn.Sequential(
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 2))
        self.fc_d = nn.Sequential(
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),

            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 2)
        )
        #  is_fire is_scope is_jump is_crouch is_walking is_reload is_e switch
        self.fc_fire = nn.Sequential(
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 2))
        self.fc_scope = nn.Sequential(
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 2))
        self.fc_jump = nn.Sequential(
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 2))
        self.fc_crouch = nn.Sequential(
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 2))
        self.fc_walking = nn.Sequential(
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 2))
        self.fc_reload = nn.Sequential(
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 2))
        self.fc_e = nn.Sequential(
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 2))
        self.fc_switch = nn.Sequential(
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 6))
        self.somftmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # print("x in put shape", x.shape)  # [406, 3, 170, 128]
        # print("x in put shape", x.shape)  # [406, 16, 3, 170, 128]
        x = x.unsqueeze(0)
        xs = torch.unbind(x, dim=1)  # [406, 16, 3, 170, 128] -> 16 x [406, 3, 170, 128]
        print("x shape ", x.shape)
        # torch.Size([50, 3, 170, 128])
        x = torch.stack([self.efficient_net.extract_features(each) for each in xs],
                        dim=1)  # 16 x [406, 1280, 5, 4] ->  [406, 16, 1280, 5, 4]
        # print("feature shape after efficient", x.shape)  # ([406, 1280, 5, 4])
        # ([406, 16, 1280, 5, 4]) # B, T, C, H, W
        x = x.float()
        layer_output_list, last_state_list = self.convlstm(x)
        x = layer_output_list[-1].half()  # [406, 1, 128, 5, 4] this is the last hidden layer's output #.squeeze()
        # layer_output_list 3 1 torch.Size([406, 16, 128, 5, 4]) last_state_list 3 2 1 torch.Size([406, 128, 5, 4])
        # torch.Size([25, 16, 64, 5, 4])
        x = torch.unbind(x, dim=1)[-1]
        # print("x shape aftrer rnnlstm ", x.shape)
        # x shape aftrer rnnlstm  torch.Size([25, 16, 64, 5, 4])

        x = torch.flatten(x, 1)  # flatten all dimensions except batch # 在实际run_agent时候，这里没有把输出打平，而是
        # print("x shape after flatten", x.shape)  # flatten torch.Size([25, 20480])
        x = self.leaky_relu(self.fc1(x))  # ([time step size, 2560])
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        # multiple out put here
        w = self.fc_w(x)  # 2
        a = self.fc_a(x)
        s = self.fc_s(x)
        d = self.fc_d(x)
        fire = self.fc_fire(x)
        scope = self.fc_scope(x)
        jump = self.fc_jump(x)
        crouch = self.fc_crouch(x)
        walking = self.fc_walking(x)
        reload = self.fc_reload(x)
        e = self.fc_e(x)
        switch = self.fc_switch(x)  # 6
        aim_x = self.fc_aim_x(x)  # 389
        aim_y = self.fc_aim_y(x)  # 209
        # w [time steps, 2]
        # we dont need softmax since we're using CrossEntropy loss. see: https://stackoverflow.com/questions/55030217/shall-i-apply-softmax-before-cross-entropy
        return w, a, s, d, fire, scope, jump, \
               crouch, walking, reload, e, switch, \
               aim_x, aim_y


import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined.half())
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param