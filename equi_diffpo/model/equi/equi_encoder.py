from escnn import gspaces, nn
import torch

from escnn import gspaces, nn
import torch

# ===================================
# Equivariant Residual Block
# ===================================
class EquiResBlock(torch.nn.Module):
    """
    Basic SO(2) equivariant residual block using ESCNN R2Conv layers.
    """
    def __init__(self, group: gspaces.GSpace2D, input_channels: int, hidden_dim: int, kernel_size: int = 3, stride: int = 1, initialize: bool = True):
        super().__init__()
        
        self.group = group
        rep = self.group.regular_repr

        feat_type_in = nn.FieldType(self.group, input_channels * [rep])
        feat_type_hid = nn.FieldType(self.group, hidden_dim * [rep])

        # First convolution and activation
        self.layer1 = nn.SequentialModule(
            nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=stride, initialize=initialize),
            nn.ReLU(feat_type_hid, inplace=True),
        )

        # Second convolution
        self.layer2 = nn.SequentialModule(
            nn.R2Conv(feat_type_hid, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, initialize=initialize),
        )

        self.relu = nn.ReLU(feat_type_hid, inplace=True)

        # Optional upscaling if input/output dimensions mismatch
        self.upscale = None
        if input_channels != hidden_dim or stride != 1:
            self.upscale = nn.SequentialModule(
                nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=1, stride=stride, bias=False, initialize=initialize),
            )

    def forward(self, x: nn.GeometricTensor) -> nn.GeometricTensor:
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)

        # Add residual connection
        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        
        return self.relu(out)

# ===================================
# 2D Equivariant Encoder (Input size 76x76)
# ===================================
class EquivariantResEncoder76Cyclic(torch.nn.Module):
    """
    2D equivariant encoder for 76x76 images using SO(2) symmetry.
    """
    def __init__(self, obs_channel: int = 2, n_out: int = 128, initialize: bool = True, N=8):
        super().__init__()

        self.obs_channel = obs_channel
        self.group = gspaces.rot2dOnR2(N)

        self.conv = torch.nn.Sequential(
            nn.R2Conv(
                nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                kernel_size=5, padding=0, initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            EquiResBlock(self.group, n_out // 8, n_out // 8, initialize=initialize),
            EquiResBlock(self.group, n_out // 8, n_out // 8, initialize=initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), 2),
            EquiResBlock(self.group, n_out // 8, n_out // 4, initialize=initialize),
            EquiResBlock(self.group, n_out // 4, n_out // 4, initialize=initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), 2),
            EquiResBlock(self.group, n_out // 4, n_out // 2, initialize=initialize),
            EquiResBlock(self.group, n_out // 2, n_out // 2, initialize=initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), 2),
            EquiResBlock(self.group, n_out // 2, n_out, initialize=initialize),
            EquiResBlock(self.group, n_out, n_out, initialize=initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out * [self.group.regular_repr]), 3),
            nn.R2Conv(
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                kernel_size=3, padding=0, initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
        )

    def forward(self, x) -> nn.GeometricTensor:
        if isinstance(x, torch.Tensor):
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x)

# ===================================
# 3D Equivariant Encoders for Voxels
# ===================================

# ---- Encoder for 58x58x58 input ----
class EquivariantVoxelEncoder58Cyclic(torch.nn.Module):
    """
    3D equivariant encoder for voxel grids of size 58^3.
    """
    def __init__(self, obs_channel: int = 4, n_out: int = 128, initialize: bool = True, N=8):
        super().__init__()

        self.obs_channel = obs_channel
        self.group = gspaces.rot2dOnR3(N)

        self.conv = torch.nn.Sequential(
            nn.R3Conv(nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                      nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]), 2),
            nn.R3Conv(nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            nn.R3Conv(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), 2),
            nn.R3Conv(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), inplace=True),
            nn.R3Conv(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), 2),
            nn.R3Conv(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), inplace=True),
            nn.R3Conv(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), 2),
            nn.R3Conv(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
        )

    def forward(self, x) -> nn.GeometricTensor:
        if isinstance(x, torch.Tensor):
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x)

# ---- Encoder for 64x64x64 input and 16x16x16 input ----
# - Update padding/kernel_size as needed for voxel size.
    
class EquivariantVoxelEncoder64Cyclic(torch.nn.Module):
    def __init__(self, obs_channel: int = 4, n_out: int = 128, initialize: bool = True, N=8):
        super().__init__()
        self.obs_channel = obs_channel
        self.group = gspaces.rot2dOnR3(N)
        self.conv = torch.nn.Sequential(
            # 64
            nn.R3Conv(
                nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]),
                kernel_size=3,
                padding=1,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]), 2),
            # 32
            nn.R3Conv(nn.FieldType(self.group, n_out // 16 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            nn.R3Conv(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), 2),
            # 16
            nn.R3Conv(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), inplace=True),
            nn.R3Conv(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), 2),
            # 8
            nn.R3Conv(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), inplace=True),
            nn.R3Conv(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), inplace=True),
            # 6
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), 2),
            # 3
            nn.R3Conv(
                nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                kernel_size=3,
                padding=0,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
            # 1x1
        )
    def forward(self, x) -> nn.GeometricTensor:
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x)
    
class EquivariantVoxelEncoder16Cyclic(torch.nn.Module):
    def __init__(self, obs_channel: int = 4, n_out: int = 128, initialize: bool = True, N=8):
        super().__init__()
        self.obs_channel = obs_channel
        self.group = gspaces.rot2dOnR3(N)
        self.conv = torch.nn.Sequential(
            # 16
            nn.R3Conv(
                nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                kernel_size=3,
                padding=1,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), 2),
            # 8
            nn.R3Conv(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            # 6
            nn.ReLU(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), inplace=True),
            nn.R3Conv(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool3D(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), 2),
            # 3
            nn.R3Conv(
                nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                kernel_size=3,
                padding=0,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
            # 1x1
        )
    def forward(self, x) -> nn.GeometricTensor:
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x)