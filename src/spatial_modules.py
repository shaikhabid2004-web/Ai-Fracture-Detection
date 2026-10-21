import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (part of CBAM).
    Highlights fracture-sensitive feature channels while suppressing background noise.
    """
    def __init__(self, in_planes, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced_planes = max(in_planes // reduction_ratio, 8)
        self.fc1 = nn.Conv2d(in_planes, reduced_planes, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (part of CBAM).
    Aggregates channel statistics across each pixel and applies a 7x7 conv
    to produce a 2D spatial attention mask focusing on fine cortical disruption.
    Uses replicate padding to avoid boundary edge artifacts in radiographs.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, padding_mode='replicate', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module: Channel Attention + Spatial Attention.
    """
    def __init__(self, in_planes, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class InterBoneRelationalAttention(nn.Module):
    """
    Inter-Bone Relational Attention Module.
    Models spatial correlation between parallel adjacent bones (e.g. Tibia & Fibula, Metatarsals).
    Uses scaled dot-product attention to prevent gradient saturation across interosseous spaces.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.proj_dim = max(in_channels // 4, 32)
        self.query_conv = nn.Conv2d(in_channels, self.proj_dim, 1)
        self.key_conv = nn.Conv2d(in_channels, self.proj_dim, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.scale = self.proj_dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        
        # Project queries, keys, values
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1) # B x (H*W) x C'
        proj_key = self.key_conv(x).view(B, -1, H * W)                      # B x C' x (H*W)
        
        # Compute scaled spatial relational correlation matrix
        energy = torch.bmm(proj_query, proj_key) / self.scale               # B x (H*W) x (H*W)
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value_conv(x).view(B, -1, H * W)                  # B x C x (H*W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        # Residual connection scaled by learnable parameter gamma
        out = self.gamma * out + x
        return out

class DualViewCrossAttention(nn.Module):
    """
    Dual-View Cross-Attention Module (AP + Lateral 3D Fusion).
    Enables Antero-Posterior (AP) and Lateral radiographic views to cross-attend to each other,
    verifying cortical fracture disruption across complementary orthogonal projections.
    """
    def __init__(self, in_channels=512, reduction=4):
        super().__init__()
        self.in_channels = in_channels
        inter_channels = max(in_channels // reduction, 32)
        
        # Projections for AP -> Lateral cross attention
        self.q_ap = nn.Conv2d(in_channels, inter_channels, 1)
        self.k_lat = nn.Conv2d(in_channels, inter_channels, 1)
        self.v_lat = nn.Conv2d(in_channels, in_channels, 1)
        
        # Projections for Lateral -> AP cross attention
        self.q_lat = nn.Conv2d(in_channels, inter_channels, 1)
        self.k_ap = nn.Conv2d(in_channels, inter_channels, 1)
        self.v_ap = nn.Conv2d(in_channels, in_channels, 1)
        
        # Learnable fusion scaling parameters
        self.gamma_ap = nn.Parameter(torch.zeros(1))
        self.gamma_lat = nn.Parameter(torch.zeros(1))
        
        # Fusion projection back to in_channels
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_ap, feat_lat):
        B, C, H, W = feat_ap.size()
        N = H * W
        
        # 1. AP queries attend to Lateral keys & values
        q_ap = self.q_ap(feat_ap).view(B, -1, N).permute(0, 2, 1) # B x N x C'
        k_lat = self.k_lat(feat_lat).view(B, -1, N)                 # B x C' x N
        v_lat = self.v_lat(feat_lat).view(B, -1, N)                 # B x C x N
        
        scale = (C ** 0.5)
        energy_ap = torch.bmm(q_ap, k_lat) / scale                  # B x N x N
        attn_ap = F.softmax(energy_ap, dim=-1)
        cross_ap = torch.bmm(v_lat, attn_ap.permute(0, 2, 1)).view(B, C, H, W)
        out_ap = feat_ap + self.gamma_ap * cross_ap
        
        # 2. Lateral queries attend to AP keys & values
        q_lat = self.q_lat(feat_lat).view(B, -1, N).permute(0, 2, 1) # B x N x C'
        k_ap = self.k_ap(feat_ap).view(B, -1, N)                    # B x C' x N
        v_ap = self.v_ap(feat_ap).view(B, -1, N)                    # B x C x N
        
        energy_lat = torch.bmm(q_lat, k_ap) / scale                 # B x N x N
        attn_lat = F.softmax(energy_lat, dim=-1)
        cross_lat = torch.bmm(v_ap, attn_lat.permute(0, 2, 1)).view(B, C, H, W)
        out_lat = feat_lat + self.gamma_lat * cross_lat
        
        # 3. Fuse orthogonal cross-attended representations
        fused = torch.cat([out_ap, out_lat], dim=1)
        out_fused = self.fusion_conv(fused)
        
        return out_fused, out_ap, out_lat

