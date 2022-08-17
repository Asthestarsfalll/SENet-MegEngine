import megengine as mge
import megengine.functional as F
import megengine.module as M


class MaxPool2d(M.MaxPool2d):
    """
        Just for SENet.
    """

    def __init__(self, ceil_mode=False, **kwargs):
        self.ceil_mode = ceil_mode
        super(MaxPool2d, self).__init__(**kwargs)

    def forward(self, inp):
        if self.ceil_mode:
            inp = F.nn.pad(inp, ((0, 0), (0, 0), (0, 1), (0, 1)))
        return super().forward(inp)
