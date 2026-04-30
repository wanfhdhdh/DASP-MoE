import torch.nn as nn


class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kH//2] = 0
        # if kH == 5:
        #     self.mask[:, :, 1:-1, 1:-1] = 0
        # else:
        #     pass
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class ColMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class RowMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, :, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class fSzMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, :, kH // 2] = 0
        self.mask[:, :, kW // 2, :] = 0


    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class SzMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(0)
        self.mask[:, :, :, kH // 2] = 1
        self.mask[:, :, kW // 2, :] = 1
        self.mask[:, :, kW // 2, kH // 2] = 0


    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class angle135MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, i, i] = 0
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class angle45MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, kW -1-i, i] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class chaMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(0)
        for i in range(kH):
            self.mask[:, :, i, i] = 1
            self.mask[:, :, kW - 1 - i, i] = 1
            self.mask[:, :, kH // 2, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class fchaMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, i, i] = 0
            self.mask[:, :, kW -1-i, i] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class huiMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, 1:-1, 1:-1] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


# ============ 新增的大核掩膜类 ============

class DeepHuiMaskedConv2d(nn.Conv2d):
    """深层回字掩膜 - 用于7×7和9×9的上分支"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        if kH == 7:
            self.mask.fill_(1)
            self.mask[:, :, 2:5, 2:5] = 0
        elif kH == 9:
            self.mask.fill_(1)
            self.mask[:, :, 2:7, 2:7] = 0
        else:
            raise ValueError(f"DeepHuiMaskedConv2d只支持7×7和9×9，当前为{kH}")
    
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class ThickAngle45MaskedConv2d(nn.Conv2d):
    """粗斜线掩膜 - 用于7×7和9×9的下分支"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            for offset in [-1, 0, 1]:
                j = kW - 1 - i + offset
                if 0 <= j < kW:
                    self.mask[:, :, kW - 1 - i, j] = 0
    
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class SuperHuiMaskedConv2d(nn.Conv2d):
    """超深回字掩膜 - 9×9的增强版"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        if kH != 9:
            raise ValueError(f"SuperHuiMaskedConv2d只支持9×9，当前为{kH}")
        self.mask.fill_(1)
        self.mask[:, :, 2:7, 2:7] = 0
    
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)