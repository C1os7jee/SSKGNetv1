import torch
import torch.nn as nn
import os

# 支持绝对导入
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from pvt_v2 import pvt_v2_b2
from swin_v1 import swin_v1_l
from stripnet import StripNet
from config import Config

def check_state_dict(state_dict, unwanted_prefixes=['module.', '_orig_mod.']):
    for k, v in list(state_dict.items()):
        prefix_length = 0
        for unwanted_prefix in unwanted_prefixes:
            if k[prefix_length:].startswith(unwanted_prefix):
                prefix_length += len(unwanted_prefix)
        if prefix_length > 0:
            state_dict[k[prefix_length:]] = state_dict.pop(k)
    return state_dict

def load_weights(model, model_name, config):

    weights_path = config.weights.get(model_name)
    if not weights_path or not os.path.exists(weights_path):
        print(f"Warning: Pretrained weights for {model_name} not found at {weights_path}. Skipping weight loading.")
        return model

    save_model = torch.load(weights_path, map_location='cpu')
    if hasattr(save_model, 'state_dict'):
        save_model = save_model.state_dict()

    candidate_dicts = []
    if isinstance(save_model, dict):
        candidate_dicts.append(save_model)
        for key in ['state_dict', 'model', 'module', 'net']:
            sub_dict = save_model.get(key)
            if isinstance(sub_dict, dict):
                candidate_dicts.append(sub_dict)
        if len(save_model) == 1:
            only_value = next(iter(save_model.values()))
            if isinstance(only_value, dict):
                candidate_dicts.append(only_value)
    else:
        print(f"Warning: Unexpected checkpoint format for {model_name}.")
        return model

    model_dict = model.state_dict()
    matched_state = {}
    for cand in candidate_dicts:
        cand_checked = check_state_dict(dict(cand))
        state_dict = {k: v for k, v in cand_checked.items() if k in model_dict and v.size() == model_dict[k].size()}
        if state_dict:
            matched_state = state_dict
            break

    if not matched_state:
        print(f"Warning: Could not load any matching weights for {model_name}. Check the state dict keys.")
        return model

    model_dict.update(matched_state)
    model.load_state_dict(model_dict)
    print(f"Successfully loaded {len(matched_state)} weights for {model_name}.")
    return model

def build_encoder(pretrained=True):
    config = Config()
    bb_name = config.bb

    def build_stripnet(cfg):
        sn_cfg = cfg.stripnet_cfg
        return StripNet(
            in_chans=cfg.model_in_channels,
            embed_dims=sn_cfg['embed_dims'],
            mlp_ratios=sn_cfg['mlp_ratios'],
            depths=sn_cfg['depths'],
            k1s=sn_cfg['k1s'],
            k2s=sn_cfg['k2s'],
            drop_path_rate=sn_cfg.get('drop_path_rate', 0.0),
        )

    backbone_builders = {
        'pvt_v2_b2': lambda cfg: pvt_v2_b2(in_channels=cfg.model_in_channels),
        'swin_v1_l': lambda cfg: swin_v1_l(in_channels=cfg.model_in_channels),
        'stripnet_s': build_stripnet,
    }

    if bb_name not in backbone_builders:
        raise NameError(f"Backbone '{bb_name}' is not defined in encoder.py")

    bb = backbone_builders[bb_name](config)

    if pretrained:
        bb = load_weights(bb, bb_name, config)

    return bb
