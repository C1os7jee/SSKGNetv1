
import os

class Config():
    def __init__(self) -> None:
        # 定义输入通道数 (3 for RGB, 5 for RGB+Gradients)
        self.model_in_channels = 3

        # 定义骨干网络，可通过环境变量覆盖
        # 可选项: 'pvt_v2_b2', 'swin_v1_l', 'stripnet_s' 等
        self.bb = os.getenv('GDR_BACKBONE', 'pvt_v2_b2')

        # 定义预训练权重路径
        # 注意：您需要确保这些权重文件存在于指定的路径下
        # 我将使用BiRefNet中定义的路径作为示例
        self.weights_root_dir = os.path.expanduser('~/CJ-Seg/Models/BiRefNet/weights/cv')
        self.weights = {
            'pvt_v2_b2': '/home/ubuntu/CJ-Seg/GDR_Net/weight/pvt_v2_b2.pth',
            'swin_v1_l': os.path.join(self.weights_root_dir, 'swin_large_patch4_window12_384_22k.pth'),
            'stripnet_s': '/home/ubuntu/CJ-Seg/GDR_Net_2/weight/stripnet_s.pth',
        }

        # 编码器输出的通道数，这取决于所选的骨干网络
        self.encoder_channels = {
            'pvt_v2_b2': [64, 128, 320, 512],
            'swin_v1_l': [192, 384, 768, 1536],
            'stripnet_s': [64, 128, 256, 512],
        }[self.bb]

        # StripNet 的默认结构配置（如需切换不同版本可在此修改）
        self.stripnet_cfg = {
            'embed_dims': (64, 128, 256, 512),
            'mlp_ratios': (8, 8, 4, 4),
            'depths': (3, 4, 6, 3),
            'k1s': (1, 1, 1, 1),
            'k2s': (19, 19, 19, 19),
            'drop_path_rate': 0.1,
        }

        # 解码器期望的输入通道数 (与编码器输出反向匹配)
        self.decoder_dims = self.encoder_channels[::-1]
        self.decoder_type = os.getenv('GDR_DECODER', 'cka')
        self.dynamic_strip = os.getenv('GDR_DYNAMIC_STRIP', '0') == '1'
        # 条状卷积每个阶段的 k2（大到小），默认 [19,15,11,7]，可用环境变量 STRIP_K2S 覆盖，格式如 "19,15,11,7"
        env_k2s = os.getenv('STRIP_K2S')
        if env_k2s:
            try:
                self.strip_k2s = tuple(int(x) for x in env_k2s.split(','))
            except Exception:
                self.strip_k2s = (19, 15, 11, 7)
        else:
            self.strip_k2s = (19, 15, 11, 7)
        # 是否在浅层 (Stage2/1) 注入 GCN，可用环境变量 USE_LOW_GCN=1/0 控制
        self.use_low_gcn = os.getenv('USE_LOW_GCN', '0') == '1'

        # GCN分支的输出通道数
        self.gcn_channels = 64

        # 添加缺失的配置项
        self.SDPA_enabled = False
