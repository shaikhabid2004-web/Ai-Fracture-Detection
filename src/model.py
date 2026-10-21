import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.spatial_modules import CBAM, InterBoneRelationalAttention, DualViewCrossAttention

FRACTURE_SUBTYPES = [
    "Hairline / Fissure (Minor crack)",
    "Transverse / Oblique (Linear break)",
    "Comminuted (Multi-fragment)",
    "Greenstick (Incomplete bending)",
    "Dislocation / Severe Displacement"
]

class DualBoneFractureNetwork(nn.Module):
    """
    Upgraded Dual-View Bone Fracture Network with:
    - Pretrained ResNet18 feature extractor adapted for 1-channel radiographs
    - CBAM Spatial Attention to highlight fine cortical bone disruption
    - Inter-Bone Relational Attention to model spatial dependencies between parallel bones
    - Dual-View Cross-Attention (AP + Lateral 3D Orthogonal Fusion)
    - Multi-Task Clinical Heads: Binary Fracture Detection + Subtype Classification
    - Grad-CAM++ hook support for localization
    """
    def __init__(self, pretrained=True, num_classes=1, num_subtypes=5):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
        
        # Adapt conv1 for single-channel grayscale X-ray
        orig_conv1 = base.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(orig_conv1.weight.mean(dim=1, keepdim=True))
                
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        
        self.layer1 = base.layer1 # 64 ch
        self.layer2 = base.layer2 # 128 ch
        self.layer3 = base.layer3 # 256 ch
        self.cbam = CBAM(in_planes=256, reduction_ratio=16, kernel_size=7)
        
        self.layer4 = base.layer4 # 512 ch
        self.inter_bone_att = InterBoneRelationalAttention(in_channels=512)
        
        # Dual-View Cross-Attention (AP + Lateral 3D Fusion)
        self.cross_view_att = DualViewCrossAttention(in_channels=512, reduction=4)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Shared projection
        self.shared_fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True)
        )
        
        # Multi-task heads:
        # Head 1: Binary Fracture Detection (Logit for Sigmoid)
        self.detection_head = nn.Linear(128, num_classes)
        # Head 2: Fracture Subtype Classification (5 clinical classes)
        self.subtype_head = nn.Linear(128, num_subtypes)
        # Head 3: Dedicated Crack Localization Head (Predicts 64x64 continuous crack heatmap)
        self.loc_head = CrackLocalizationHead(in_channels_l3=256, in_channels_l4=512)
        
        # Hooks for Grad-CAM++
        self.gradients = None
        self.activations = None
        
    def hook_activations(self, module, input, output):
        self.activations = output

    def hook_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]
        
    def register_cam_hooks(self, target_layer=None):
        # Target high-resolution Layer 3 (32x32) to capture fine cortical bone disruption
        if target_layer is None:
            target_layer = self.layer3[1].conv2
        target_layer.register_forward_hook(self.hook_activations)
        target_layer.register_full_backward_hook(self.hook_gradients)

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        f3 = self.layer3(x)
        f3 = self.cbam(f3)
        
        f4 = self.layer4(f3)
        f4 = self.inter_bone_att(f4)
        return f3, f4

    def forward(self, x_ap, x_lat=None, return_subtypes=True, return_localization=False):
        # Handle dual-view input or single-view / landscape splitting
        is_dual = False
        if x_lat is not None:
            is_dual = True
            x_ap_view = x_ap
            x_lat_view = x_lat
        elif x_ap.size(-1) >= int(x_ap.size(-2) * 1.5):
            is_dual = True
            mid = x_ap.size(-1) // 2
            x_ap_view = x_ap[:, :, :, :mid]
            x_lat_view = x_ap[:, :, :, mid:]
        else:
            x_ap_view = x_ap
            
        if is_dual:
            f3_ap, f4_ap = self.extract_features(x_ap_view)
            f3_lat, f4_lat = self.extract_features(x_lat_view)
            fused_feat, _, _ = self.cross_view_att(f4_ap, f4_lat)
            f3_fused = (f3_ap + f3_lat) * 0.5
        else:
            f3_fused, fused_feat = self.extract_features(x_ap_view)
            
        features = self.avgpool(fused_feat)
        features = torch.flatten(features, 1)
        shared = self.shared_fc(features)
        
        detection_logits = self.detection_head(shared)
        subtype_logits = self.subtype_head(shared)
        loc_heatmap = self.loc_head(f3_fused, fused_feat)
        
        if return_localization:
            if return_subtypes:
                return detection_logits, subtype_logits, loc_heatmap
            return detection_logits, loc_heatmap
            
        if return_subtypes:
            return detection_logits, subtype_logits
        return detection_logits

class CrackLocalizationHead(nn.Module):
    """
    Dedicated Saliency & Crack Localization Head.
    Fuses Multi-Scale Feature Maps (Layer 3 + Layer 4) with Spatial Attention
    to predict precise pixel-level fracture probability heatmaps.
    """
    def __init__(self, in_channels_l3=256, in_channels_l4=512):
        super().__init__()
        self.conv_l4 = nn.Sequential(
            nn.Conv2d(in_channels_l4, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.conv_l3 = nn.Sequential(
            nn.Conv2d(in_channels_l3, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 32x32
            nn.Conv2d(128, 64, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 64x64
            nn.Conv2d(64, 1, kernel_size=3, padding=1, padding_mode='replicate')
        )

    def forward(self, feat_l3, feat_l4):
        p4 = self.conv_l4(feat_l4)
        p3 = self.conv_l3(feat_l3)
        fused = torch.cat([p3, p4], dim=1)
        return self.decoder(fused)

if __name__ == "__main__":
    net = DualBoneFractureNetwork(pretrained=False)
    net.register_cam_hooks()
    dummy_ap = torch.randn(2, 1, 512, 512)
    dummy_lat = torch.randn(2, 1, 512, 512)
    det, sub = net(dummy_ap, dummy_lat)
    print("DualBoneFractureNetwork dual input shapes -> Det:", det.shape, "Subtype:", sub.shape)
    
    dummy_landscape = torch.randn(2, 1, 512, 1024)
    det2, sub2 = net(dummy_landscape)
    print("DualBoneFractureNetwork landscape auto-split -> Det:", det2.shape, "Subtype:", sub2.shape)

