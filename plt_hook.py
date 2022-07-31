import torch, math, time, copy
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class PltAboutHook(object):
    """
    The library is easy to use, just initialize it before the model is trained and then call the show() method at the right time.
    """

    def __init__(self, model, type='forward', norm='mean', log=False, hook_function=None):
        """
        Args:
            model: nn.Module(), It can be any model with nn.Module as base class, which will only be registered with hooks in the model's minimal leaf operator.
            type: str, one of forward and backward
            norm: str, one of sum and mean
            log: bool, since the norm of some parameters may change drastically, the log can be set to True for better observation
            hook_function: callable, you can override the hook_function defined in this class
        """
        super(PltAboutHook, self).__init__()
        self.model = model
        if hook_function != None:
            self.hook_function = hook_function
        assert type in ['backward', 'forward'], "type should be one of backward and forward"
        self.type = type
        self.buffer_dict = {}
        self.count = 0
        self.norm = norm
        self.log = log
        self.all_for_registed_hook(model)

    def registed_hook(self, module: nn.Module, hook_function=None):
        hook_function = hook_function if hook_function != None else self.hook_function
        if self.type == 'forward':
            module.register_forward_hook(hook_function)
        else:
            module.register_backward_hook(hook_function)

    def hook_function(self, module: nn.Module, inputs, outputs):
        if module != nn.Identity():
            if str(module) + str(id(module)) not in self.buffer_dict:
                self.buffer_dict[str(module) + str(id(module))] = {}
                if not isinstance(inputs, torch.Tensor):
                    self.buffer_dict[str(module) + str(id(module))]['inputs'] = [[] for i in range(len(inputs))]
                else:
                    self.buffer_dict[str(module) + str(id(module))]['inputs'] = [[]]
                if not isinstance(inputs, torch.Tensor):
                    self.buffer_dict[str(module) + str(id(module))]['outputs'] = [[] for i in range(len(outputs))]
                else:
                    self.buffer_dict[str(module) + str(id(module))]['outputs'] = [[]]
            if self.norm == 'mean':
                for i, input in enumerate(inputs):
                    self.buffer_dict[str(module) + str(id(module))]['inputs'][i].append(
                        (self.count, (input.norm() / input.numel()).detach().item()))
                for i, output in enumerate(outputs):
                    self.buffer_dict[str(module) + str(id(module))]['outputs'][i].append(
                        (self.count, (output.norm() / output.numel()).detach().item()))
            else:
                for i, input in enumerate(inputs):
                    self.buffer_dict[str(module) + str(id(module))]['inputs'][i].append(
                        (self.count, input.norm().detach().item()))
                for i, output in enumerate(outputs):
                    self.buffer_dict[str(module) + str(id(module))]['outputs'][i].append(
                        (self.count, output.norm().detach().item()))
            self.count += 1

    def all_for_registed_hook(self, modules):
        for module in modules.children():
            if len(list(module.children())) > 0:
                self.all_for_registed_hook(module)
            elif len(list(module.children())) <= 0 and len(str(module)) < 100:
                self.registed_hook(module, hook_function=self.hook_function)

    def zys(self, n):
        value = []
        i = 2
        m = n
        while i <= int(m / 2 + 1) and n != 1:
            if n % i == 0:
                value.append(i)
                n = n // i
                i -= 1
            i += 1
        value.append(1)
        if len(value) == 1:
            value.append(m)
        l1 = 1
        l2 = 1
        tag = 0
        for i in value:
            if tag:
                l1 *= i
                tag = 0
            else:
                l2 *= i
                tag = 1
        if max(l1, l2) / min(l1, l2) > 2:
            l1 = l2 = int(math.sqrt(l1 * l2) + 1)
        return (l2, l1)

    def clear(self):
        self.buffer_dict.clear()

    def init_figure(self):
        l = 2
        a, b = self.x_y_
        self.fig = plt.figure(figsize=(l * a, l * b))
        self.subfigurecount = 0
        self.format_font = \
            {
                'family': "Times New Roman",
                'weight': 'bold',
                'style': 'normal',
                'size': 6,
            }
        plt.rc('font', **self.format_font)
        plt.subplots_adjust(wspace=0.1, hspace=0.5, right=1, bottom=0, top=1, left=0)

    def get_subfigure(self):
        sub = self.fig.add_subplot(self.x_y_[0], self.x_y_[1], self.subfigurecount + 1)
        self.subfigurecount += 1
        return sub

    def show(self):

        # TODO: 子图数量就是总的字典键值
        names = self.buffer_dict.keys()
        n = len(names)
        self.x_y_ = self.zys(n)
        self.init_figure()

        # TODO: 同一字典键值的所有输入和输出显示在一张图像中
        for name in names:
            ax = self.get_subfigure()
            value = self.buffer_dict[name]
            inputs = value['inputs']
            outputs = value['outputs']
            cname:str=name[:name.rfind(')') + 1]
            if len(cname) > 30:
                cname = cname[:int(len(cname)//2)] + '\n' + cname[int(len(cname)//2):]
            ax.set_title(cname, {'style': 'normal', 'size': 6})
            ax.patch.set_facecolor("white")
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_facecolor('white')
            x = np.arange(len(inputs[0]))
            # TODO: 循环显示
            for input in inputs:
                y = [sy[1] if not self.log else math.log(sy[1]) for sy in input]
                ax.plot(x, y, '--', linewidth=2, linestyle='dashed')
            for output in outputs:
                y = [sy[1] if not self.log else math.log(sy[1]) for sy in output]
                ax.plot(x, y, '-', linewidth=2, linestyle='dashed')
            ax.set_xticks([])
        plt.xticks([])
        plt.savefig(str(id(self.model)) + '_' + str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())) + '.png')
        plt.show()
