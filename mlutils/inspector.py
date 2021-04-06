import cv2
import numpy as np
import torch


class ModuleNode(object):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name
        self.children = {}
    
    def remove_child(self, name):
        del self.children[name]

    @property
    def empty(self):
        return self.isleaf

    @property
    def isleaf(self):
        return len(self.children) == 0

    def has_child(self, name):
        return name in self.children

    def append_child(self, child):
        self.children[child.name] = child

    def get_child(self, name):
        return self.children[name]

    def __str__(self):
        root = 'root'
        return f'{self.name or root}, number of children {len(self.children)}'


class ModuleForest(ModuleNode):
    def __init__(self, name=None) -> None:
        super().__init__(name)
        self.children = {}

    def add_layer(self, layer_name):
        splited_name = layer_name.split('.')
        node = self
        for name in splited_name:
            if node.has_child(name):
                node = node.get_child(name)
            else:
                new_node = ModuleNode(name)
                node.append_child(new_node)
                node = new_node


def one_hot(inp, num_classes):
    output = torch.zeros((inp.shape[0], num_classes, *inp.shape[1:])).to(inp.device)
    output.scatter_(1, inp.unsqueeze(1), 1)
    return output


class Inspector(object):
    '''
        >>> # init
        >>> model = Model()
        >>> inspector = Inspector(model)
        >>> inspector.regist_layers(['35'])
        >>> inspector.regist_loss_fn(loss_fn)
        >>> # run
        >>> inspector.inspect(images, labels)
        >>> cam = inspected_model.show_cam_on_image()
    '''
    def __init__(self, opt, model) -> None:
        super().__init__()
        self.opt = opt
        self.model = model
        self.one_hot = one_hot
        self.regist_module_forest = ModuleForest()
        self.gradients = []
        self.features = []
        self.cams = []
        self.image = None
        self.training = False

    def regist_layers(self, *layers):
        for layer in layers:
            self.regist_module_forest.add_layer(layer)
    
    def regist_one_hot(self, one_hot):
        self.one_hot = one_hot

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def inspect(self, x, y=None):
        assert not self.regist_module_forest.empty, \
            'No layer has been registed yet.'
        assert self.one_hot is not None, \
            'one_hot function is None'
        if self.model.training:
            self.training = True
        self.model.eval()
        self.model.zero_grad()
        self.image = x
        self.features = []
        self.cams = []
        self.gradients = []

        def exec(module, node, x):
            if node.isleaf:
                x = module(x)
                x.register_hook(self.save_gradient)
                self.features += [x]
                return x

            for name, m in module.named_children():
                if not node.has_child(name):
                    # TODO fix dirty code
                    if name in ['classifier', 'fc'] and x.ndim > 2:
                        x = x.view(x.size(0), -1)
                    x = m(x)
                else:
                    child_node = node.get_child(name)
                    x = exec(m, child_node, x)
            return x

        x = exec(self.model, self.regist_module_forest, x)
        if y is None:
            y = torch.argmax(x).unsqueeze(0)

        y = self.one_hot(inp=y, num_classes=x.size(1))
        loss = torch.sum(x * y)
        loss.backward(retain_graph=True)

        for feat, grad in zip(self.features, self.gradients):
            # get first image of the batch
            feat = feat[0]
            grad = grad[0]
            self.image = self.image[0].detach()

            cam = torch.zeros(feat.shape[-2:])
            weights = torch.mean(grad, dim=(1, 2))
            for i, w in enumerate(weights):
                cam += feat[i, :, :] * w
            self.cams.append(cam)

        if self.training:
            self.model.train()

    def show_cam_on_images(self, image):
        assert len(self.cams) > 0, 'NO cam to show.'
        outputs = []
        if image is None:
            image = self.image

        if isinstance(image, torch.Tensor):
            image = image.numpy()

        for cam in self.cams:
            if isinstance(cam, torch.Tensor):
                cam = cam.detach().cpu().numpy()

            # adjust cam to 0~1
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, image.shape[-2:])
            cam = cam - cam.min()
            cam = cam / cam.max()
            heatmap = cv2.applyColorMap(np.uint8(255 * cam),
                                        cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255.
            # image = (image - image.min()) / (image.max() - image.min())
            cam_image = heatmap.transpose((2, 0, 1))[::-1, :, :] \
                        + np.float32(image)
            cam_image = cam_image / cam_image.max()
            outputs.append(cam_image)

        self.cams = []
        self.features = []
        self.gradients = []
        return outputs
