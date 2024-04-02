import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRF(nn.Module):
    def __init__(self,
                input_ch=3,
                layer_ch=8,
                filter_ch=255,
                skip=[4],
                use_viewdirs=False):
        self.input_ch = input_ch
        self.skip = skip
        self.act = F.relu
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
        [nn.Linear(self.d_input, filter_ch)] +
        [nn.Linear(filter_ch + self.d_input, filter_ch) if i in skip \
        else nn.Linear(filter_ch, filter_ch) for i in range(layer_ch - 1)]
        )

        # Bottleneck
        if self.d_viewdirs is not None:
        #分离alpha和RGB
            self.alpha_out = nn.Linear(filter_ch, 1)
            self.rgb_filters = nn.Linear(filter_ch, filter_ch)
            self.branch = nn.Linear(filter_ch + self.d_viewdirs, filter_ch // 2)
            self.output = nn.Linear(filter_ch // 2, 3)
        else:
        #or output the simple results
            self.output = nn.Linear(filter_ch, 4)
    
    def forward(
        self,
        x: torch.Tensor,
        viewdirs: Optional[torch.Tensor] = None
    ) -> torch.Tensor: