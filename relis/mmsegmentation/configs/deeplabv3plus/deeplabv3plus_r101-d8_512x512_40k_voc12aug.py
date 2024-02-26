_base_ = './deeplabv3plus_r50-d8_512x512_40k_voc12aug.py'

pretrained_resnet101v1c = '/gfs-ssd/project/uss/pre_trained_models/ResNetv1c/resnetv1c101_8xb32_in1k_20220214-434fe45f.pth'
model = dict(pretrained=None,
             backbone=dict(init_cfg=dict(
                                type='Pretrained',
                                checkpoint=pretrained_resnet101v1c,
                                prefix='backbone.'
                                ),
                           depth=101,
                         )
            )
