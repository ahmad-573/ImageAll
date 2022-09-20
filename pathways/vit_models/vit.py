
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.helpers import *
from timm.models import create_model
from .transformer_block import *

from .build import MODEL_REGISTRY
from .prompt_models_196 import resPromptVisionTransformer, resPromptDino, resPromptClip
from .prompt_models_196_cat_prompt import resPromptVisionTransformer as catPromptVisionTransformer
from .prompt_models_196_baseline import resPromptVisionTransformer as baselinePromptVisionTransformer
from .prompt_models_196_baseline_linear import resPromptVisionTransformer as baselineLinearVisionTransformer
from .prompt_models import resPromptVisionTransformer as fullPromptVisionTransformer
from .vit_timesformer import VisionTransformer as TimesformerVisionTransformer
default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

__all__ = [
    'vit_base_patch16_224','vit_large_patch16_224'
]


class VanillaVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.patch_size=kwargs['patch_size']
        embed_dim=kwargs['embed_dim']
        self.depth=kwargs['depth']
        num_heads=kwargs['num_heads']
        mlp_ratio=kwargs['mlp_ratio']
        qkv_bias=kwargs['qkv_bias']
        norm_layer=kwargs['norm_layer']
        num_classes=kwargs['num_classes']
        drop_rate=kwargs['drop_rate']
        drop_path_rate=kwargs['drop_path_rate']
        attn_drop_rate=kwargs['attn_drop_rate']

        act_layer = None
        act_layer = act_layer or nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth)])

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed

        x = self.pos_drop(x)

        layer_wise_tokens = []
        attention_maps = []
        for blk in self.blocks:
            x, attn = blk(x, return_attention=True)
            layer_wise_tokens.append(x)
            attention_maps.append(attn)

        return layer_wise_tokens, attention_maps

    def forward(self, x, only_last=False,  all_tokens=False):
        layer_wise_tokens, attention_maps = self.forward_features(x)

        # Normalize features/tokens of each block
        layer_wise_tokens_norm = [self.norm(x) for x in layer_wise_tokens]
        # Apply classifier to cls token of each block
        x = [self.head(x[:,0]) for x in layer_wise_tokens_norm]
        if only_last:
            return x[-1] # return classification score of the last token
        elif all_tokens:
            return x, layer_wise_tokens, attention_maps # return classification scores of all tokens and outputs of all blocks
        else:
            return x # return classification scores of all tokens



@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    model = VanillaVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg =  _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint)
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VanillaVisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint)
    return model


############################
#       VIDEO MODELS       #
############################

## Prompt 196 original
@MODEL_REGISTRY.register()
class vit_base_patch16_224_timeP_1(nn.Module):
    def __init__(self, args, **kwargs):
        super(vit_base_patch16_224_timeP_1, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = resPromptVisionTransformer(img_size=args.img_size,num_classes=1000, actual_num_classes=args.num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=args.num_frames, attention_type='joint_space_time', **kwargs)

        self.attention_type = 'joint_space_time'
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (args.img_size // patch_size) * (args.img_size // patch_size)

        self.depth = self.model.depth

        # pretrained_model = torch.load(args.pre_trained)
        # self.model.load_state_dict(pretrained_model['model'])
        for name, param in self.model.named_parameters():
            print(name, param.shape)

    def forward(self, x, return_tokens=False):
        x_resPrompt, x_features, attention_maps = self.model(x)
        if return_tokens:
            return x_resPrompt, x_features
        else:
            return x_resPrompt

@MODEL_REGISTRY.register()
class deit_base_patch16_224_timeP_1_org(nn.Module):
    def __init__(self, args, **kwargs):
        super(deit_base_patch16_224_timeP_1_org, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = resPromptVisionTransformer(img_size=args.img_size,num_classes=1000, actual_num_classes=args.num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=args.num_frames, attention_type='joint_space_time', **kwargs)

        self.attention_type = 'joint_space_time'
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (args.img_size // patch_size) * (args.img_size // patch_size)

        self.depth = self.model.depth

        # pretrained_model = torch.load(args.pre_trained)
        # self.model.load_state_dict(pretrained_model['model'])
        for name, param in self.model.named_parameters():
            print(name, param.shape)

    def forward(self, x, return_tokens=False):
        x_resPrompt, x_features, attention_maps = self.model(x)
        if return_tokens:
            return x_resPrompt, x_features
        else:
            return x_resPrompt

@MODEL_REGISTRY.register()
class dino_base_patch16_224_1P(nn.Module):
    def __init__(self, args, **kwargs):
        super(dino_base_patch16_224_1P, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = resPromptDino(img_size=args.img_size,num_classes=1000, num_prompts=1,actual_num_classes=args.num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=args.num_frames, attention_type="joint_space_time", **kwargs)

        self.attention_type = "joint_space_time"
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (args.img_size // patch_size) * (args.img_size // patch_size)
        pretrained_model=args.pre_trained
        self.depth = self.model.depth
        # if self.pretrained:
        #     checkpoint = torch.load(pretrained_model)
        #     self.model.load_state_dict(checkpoint,strict=False)
        
        # self.model.resPrompt_token.data = self.model.cls_token.data.clone()
        
    def forward(self, x):
        x_resPrompt, _ = self.model(x)
        return x_resPrompt

@MODEL_REGISTRY.register()
class clip_base_patch16_224_1P(nn.Module):
    def __init__(self, args, **kwargs):
        super(clip_base_patch16_224_1P, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = resPromptClip(
                input_resolution = 224,
                patch_size = 16,
                width = 768,
                layers = 12,
                heads = 768 // 64,
                output_dim = 512,
                num_prompts=1,
                num_classes = 1000,
                actual_num_classes = args.num_classes,
                num_frames=args.num_frames,
                attention_type="joint_space_time")

        self.attention_type = "joint_space_time"
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (args.img_size // patch_size) * (args.img_size // patch_size)
        pretrained_model=args.pre_trained
        self.depth = self.model.depth
        # if self.pretrained:
        #     checkpoint = torch.load(pretrained_model)
        #     self.model.load_state_dict(checkpoint,strict=False)
        
        # self.model.resPrompt_token.data = self.model.class_embedding.data.unsqueeze(0).unsqueeze(0).clone()
        
    def forward(self, x):
        x_resPrompt, x_cls = self.model(x)
        return x_resPrompt


## timesformer
@MODEL_REGISTRY.register()
class timesformer_vit_base_patch16_224(nn.Module):
    def __init__(self, args, **kwargs):
        super(timesformer_vit_base_patch16_224, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = TimesformerVisionTransformer(img_size=args.img_size, num_classes=args.num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=8, attention_type="joint_space_time", **kwargs)

        self.attention_type = "joint_space_time"
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (args.img_size // patch_size) * (args.img_size // patch_size)

        self.depth = self.model.depth
        
        # for name, param in self.model.named_parameters():
        #     print(name, param.shape, param.requires_grad)

    def forward(self, x):
        x = self.model(x)
        return x

## Prompt Cat Prompt
@MODEL_REGISTRY.register()
class deit_base_patch16_224_timeP_1_cat(nn.Module):
    def __init__(self, args, **kwargs):
        super(deit_base_patch16_224_timeP_1_cat, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = catPromptVisionTransformer(img_size=args.img_size,num_classes=1000, actual_num_classes=args.num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=args.num_frames, attention_type='joint_space_time', **kwargs)

        self.attention_type = 'joint_space_time'
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (args.img_size // patch_size) * (args.img_size // patch_size)

        self.depth = self.model.depth

        # pretrained_model = torch.load(args.pre_trained)
        # self.model.load_state_dict(pretrained_model['model'])
        # for name, param in self.model.named_parameters():
        #     print(name, param.shape)

    def forward(self, x, return_tokens=False):
        x_resPrompt, x_features, attention_maps = self.model(x)
        if return_tokens:
            return x_resPrompt, x_features
        else:
            return x_resPrompt

## baseline prompt
@MODEL_REGISTRY.register()
class deit_base_patch16_224_base_prompt(nn.Module):
    def __init__(self, args, **kwargs):
        super(deit_base_patch16_224_base_prompt, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = baselinePromptVisionTransformer(img_size=args.img_size,num_classes=1000, actual_num_classes=args.num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=args.num_frames, attention_type='joint_space_time', **kwargs)

        self.attention_type = 'joint_space_time'
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (args.img_size // patch_size) * (args.img_size // patch_size)

        self.depth = self.model.depth

        # pretrained_model = torch.load(args.pre_trained)
        # self.model.load_state_dict(pretrained_model['model'])
        for name, param in self.model.named_parameters():
            print(name, param.shape)

    def forward(self, x, return_tokens=False):
        x_resPrompt, x_features, attention_maps = self.model(x)
        if return_tokens:
            return x_resPrompt, x_features
        else:
            return x_resPrompt

## baseline linear
@MODEL_REGISTRY.register()
class deit_base_patch16_224_base_lin(nn.Module):
    def __init__(self, args, **kwargs):
        super(deit_base_patch16_224_base_lin, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = baselineLinearVisionTransformer(img_size=args.img_size,num_classes=1000, actual_num_classes=args.num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=args.num_frames, attention_type='joint_space_time', **kwargs)

        self.attention_type = 'joint_space_time'
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (args.img_size // patch_size) * (args.img_size // patch_size)

        self.depth = self.model.depth

        # pretrained_model = torch.load(args.pre_trained)
        # self.model.load_state_dict(pretrained_model['model'])
        for name, param in self.model.named_parameters():
            print(name, param.shape)

    def forward(self, x, return_tokens=False):
        x_resPrompt, x_features, attention_maps = self.model(x)
        if return_tokens:
            return x_resPrompt, x_features
        else:
            return x_resPrompt

@MODEL_REGISTRY.register()
class vit_base_patch16_224_base_lin(nn.Module):
    def __init__(self, args, **kwargs):
        super(vit_base_patch16_224_base_lin, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = baselineLinearVisionTransformer(img_size=args.img_size,num_classes=1000, actual_num_classes=args.num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=args.num_frames, attention_type='joint_space_time', **kwargs)

        self.attention_type = 'joint_space_time'
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (args.img_size // patch_size) * (args.img_size // patch_size)

        self.depth = self.model.depth

        # pretrained_model = torch.load(args.pre_trained)
        # self.model.load_state_dict(pretrained_model['model'])
        for name, param in self.model.named_parameters():
            print(name, param.shape)

    def forward(self, x, return_tokens=False):
        x_resPrompt, x_features, attention_maps = self.model(x)
        if return_tokens:
            return x_resPrompt, x_features
        else:
            return x_resPrompt

# full 1568
@MODEL_REGISTRY.register()
class deit_base_patch16_224_timeP_1_full_1568(nn.Module):
    def __init__(self, args, **kwargs):
        super(deit_base_patch16_224_timeP_1_full_1568, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = fullPromptVisionTransformer(img_size=args.img_size,num_classes=1000, actual_num_classes=args.num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=args.num_frames, attention_type='joint_space_time', **kwargs)

        self.attention_type = 'joint_space_time'
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (args.img_size // patch_size) * (args.img_size // patch_size)

        self.depth = self.model.depth

        # pretrained_model = torch.load(args.pre_trained)
        # self.model.load_state_dict(pretrained_model['model'])
        for name, param in self.model.named_parameters():
            print(name, param.shape)

    def forward(self, x, return_tokens=False):
        x_resPrompt, x_feats = self.model(x)
        if return_tokens:
            return x_resPrompt, x_feats
        else:
            return x_resPrompt

if __name__=="__main__":
    model = create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
    )
    model(torch.randn(1,3,224,224))