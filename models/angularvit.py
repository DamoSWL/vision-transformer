from email.errors import ObsoleteHeaderDefect
from tokenize import group
from turtle import width

from numpy import dtype, float32
import torch
import torch.nn as nn
import timm
from timm.models.layers import trunc_normal_, DropPath,Mlp,drop_path
from timm.models.registry import register_model
from timm.models import create_model
import torch.nn.functional as F
import math
from einops import rearrange
from ptflops import get_model_complexity_info



def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp, W_sp, C)
    img_perm = img_perm.permute(0,3,1,2).contiguous()
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img
 


class PatchEmbed(nn.Module):
    def __init__(self,in_channels=3, out_channel=64,norm_layer=nn.LayerNorm):
        super(PatchEmbed, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channel//2, kernel_size=3, stride=2,padding=1),
            # nn.SyncBatchNorm(out_channel//2),
            nn.BatchNorm2d(out_channel//2),
            nn.GELU(),
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=2,padding=1),
            # nn.SyncBatchNorm(out_channel),
            nn.BatchNorm2d(out_channel),
            nn.GELU()

            )

        # self.stem = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channel, kernel_size=7, stride=4, padding=2)
        #     )
        # self.norm = norm_layer(out_channel)
        
        
    def forward(self, x):
        x = self.stem(x)
        x = x.permute(0,2,3,1).contiguous()
        # x = self.norm(x)
        return x




class ConvDownsample(nn.Module):
    def __init__(self, dim, next_dim):
        super().__init__()
        self.reduction = nn.Conv2d(dim, next_dim, kernel_size=2, stride=2, padding=0)
      
        
    def forward(self, x):    
        assert len(x.shape) == 4
        x = x.permute(0,3,1,2).contiguous()
        x = self.reduction(x)
        x = x.permute(0,2,3,1).contiguous()

        return x


class AngularAttention(nn.Module):
    def __init__(self,embed_dim=64,
                num_heads=2,
                n_window=4,
                kv_pooling=2,
                qkv_bias=True,
                attn_drop=0., 
                proj_drop=0.,
                index=0,
                attn_type='cos'):
        super(AngularAttention,self).__init__()

        self.index = index
        self.attn_type = attn_type

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.n_window = n_window
        self.kv_pooling = kv_pooling

        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(in_features=self.embed_dim, out_features=3*self.embed_dim,bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.tau = nn.Parameter(torch.ones(1,num_heads, 1, 1).fill_(0.6))


    def forward_feature(self,x):
        _,B,H,W,D = x.shape
        
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0
      
        pad = (self.n_window - H % self.n_window) % self.n_window
        pad_top = pad // 2
        pad_bottom = pad - pad_top
         
        pad = (self.n_window - W % self.n_window) % self.n_window
        pad_left = pad // 2
        pad_right = pad - pad_left

        x = F.pad(x,(0,0,pad_left,pad_right,pad_top,pad_bottom),mode='replicate')

        _,B,H1,W1,D = x.shape
        
        win_h = H1 // self.n_window
        win_w = W1 // self.n_window

        q,k,v = x[0], x[1], x[2] # B H W D

       
        q = q.permute(0,3,1,2).contiguous()
        k = k.permute(0,3,1,2).contiguous()
        v = v.permute(0,3,1,2).contiguous()

        q = img2windows(q,win_h,win_w)
        k = img2windows(k,win_h,win_w) 
        v = img2windows(v,win_h,win_w)

        q_size_H = win_h
        q_size_W = win_w
        kv_size_H = win_h
        kv_size_W = win_w



        q = q.view(-1,self.num_heads,self.head_dim,q_size_H*q_size_W).permute(0,1,3,2).contiguous()
        # (B',h,k*k,d)
        k = k.view(-1,self.num_heads,self.head_dim,kv_size_H*kv_size_W).permute(0,1,3,2).contiguous()
        v = v.view(-1,self.num_heads,self.head_dim,kv_size_H*kv_size_W).permute(0,1,3,2).contiguous()



        if self.attn_type == 'exp':
            attn = (q @ k.transpose(-2, -1)) * self.scale
     
        else:
            q = F.normalize(q,dim=-1)
            k = F.normalize(k,dim=-1)

            if self.attn_type == 'cos':
                attn = (q @ k.transpose(-2, -1)) / 0.1
            elif self.attn_type == 'quad':
                theta = torch.arccos(q @ k.transpose(-2, -1))
                attn = (1-4*(theta*theta/math.pi**2)) 
                attn /= 0.1       
            else:
                raise ValueError('unknown attention type')


        # if self.height == 14 and self.index == 0:
        #     print(theta[0,0,0,:])
        #     print(attn[0,0,0,:])
        #     exit()


        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)


        x = attn @ v # (B',h,k*k,d)
        
        x = x.permute(0,2,1,3).contiguous() 
        x = x.view(-1,win_h,win_w,D).contiguous() # (B',k,k,D)
        x = windows2img(x, win_h, win_w,H1,W1) # (B,H,W,D)
        x = x[:,pad_top:H1-pad_bottom,pad_left:W1-pad_right,:]

        return x

    def forward(self,x):
        B,H,W,D = x.shape    
           
        qkv = self.qkv(x).reshape(B, H, W, 3, self.embed_dim).permute(3, 0, 1, 2, 4).contiguous()
        x = self.forward_feature(qkv)


        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class PEG(nn.Module):
    def __init__(self, dim=256, k=3):
        super(PEG,self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim)

    def forward(self, x):
        x = rearrange(x, 'b h w d -> b d h w')
        x = self.proj(x) + x
        x = rearrange(x, 'b d h w -> b h w d')
       
        return x


class AngularLayer(nn.Module):
    def __init__(self,embed_dim=64,
                num_heads=2,
                n_window=(4,3),
                kv_pooling=2,
                attn_drop=0., 
                proj_drop=0.,
                drop_path_rate=0.,
                mlp_ratio=1.0,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU,
                index=0,
                attn_type='cos'):
        super(AngularLayer,self).__init__()

        self.index = index

        self.embed_dim = embed_dim
        self.num_heads = num_heads 

        self.peg = PEG(dim=self.embed_dim)

        self.norm_1 = norm_layer(self.embed_dim)
        self.attention = AngularAttention(embed_dim=embed_dim,num_heads=self.num_heads,
                                        n_window=n_window[self.index%2],kv_pooling=kv_pooling,
                                        attn_drop=attn_drop, proj_drop=proj_drop,index=self.index,attn_type=attn_type)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm_2 = norm_layer(self.embed_dim)

        self.mlp = Mlp(in_features=self.embed_dim,hidden_features=int(mlp_ratio * self.embed_dim),
                        act_layer=act_layer, drop=proj_drop)

    def forward(self,x):

        x = self.peg(x)
        cur = self.norm_1(x)
        x = x + self.drop_path(self.attention(cur))

        cur = self.norm_2(x)
        x = x + self.drop_path(self.mlp(cur))

        # x = x + self.drop_path(self.attention(self.norm_1(x)))
        # x = x + self.drop_path(self.mlp(self.norm_2(x)))
        return x




class AngularBlock(nn.Module):
    def __init__(self,embed_dim=64,
                next_embded_dim=64,
                depth=2,
                num_heads=2,
                n_window=(4,3),
                kv_pooling=2,
                attn_drop=0., 
                proj_drop=0.,
                drop_path_rate=0.,
                mlp_ratio=4.0,
                downsample_flag=True,
                attn_type='cos'):
        super(AngularBlock,self).__init__()

        self.embed_dim = embed_dim
        self.next_embed_dim = next_embded_dim
        self.depth = depth
        self.num_heads = num_heads
        self.downsample_flag = downsample_flag
        if self.downsample_flag:
            self.downsample = ConvDownsample(self.embed_dim,self.next_embed_dim)
        else:
            self.downsample = None


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # self.peg = PEG(dim=self.embed_dim)

        self.blocks = nn.ModuleList([AngularLayer(embed_dim=self.embed_dim,num_heads=self.num_heads,
                                                n_window=n_window,kv_pooling=kv_pooling,
                                                attn_drop=attn_drop, proj_drop=proj_drop,drop_path_rate=dpr[i],
                                                mlp_ratio=mlp_ratio,index=i,attn_type=attn_type)
            for i in range(self.depth)])

        

    def forward(self,x):

        for i,block in enumerate(self.blocks):
            x = block(x)
            # if i == 0:
            #     x = self.peg(x)

        if self.downsample_flag:
            x = self.downsample(x)

        return x




class AngularViT(nn.Module):
    def __init__(self,img_size = 224,
                num_classes=1000,
                embed_dim=[64,128,256,512],
                depth=[2,6,12,2],
                num_heads=[2,4,8,16],
                n_window=[(6,5),(4,3),(2,1),(1,1)],
                kv_pooling=[2,0,0,0],
                attn_drop_rate=0., 
                proj_drop=0.,
                drop_path_rate=0.,
                mlp_ratio=4.0,
                norm_layer=nn.LayerNorm,
                attn_type='cos',
                **kwargs):
        super().__init__()

        self.num_classes = num_classes  
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_depth = len(depth)
        self.patch_embed = PatchEmbed(in_channels=3, out_channel=self.embed_dim[0],norm_layer=norm_layer)

        self.height = img_size
        self.width = img_size

        self.height = self.height // 2**2
        self.width = self.width // 2**2

        # self.absolute_pos_embed = nn.Parameter(torch.zeros(1, (self.width)*(self.height) ,self.embed_dim[0]))

        self.stages = nn.ModuleList()
        for i in range(self.num_depth):
            single_stage = AngularBlock(embed_dim=self.embed_dim[i],depth=self.depth[i],num_heads=self.num_heads[i],
                                        n_window=n_window[i],kv_pooling=kv_pooling[i],
                                        attn_drop=attn_drop_rate, proj_drop=proj_drop,drop_path_rate=drop_path_rate,
                                        mlp_ratio=mlp_ratio,next_embded_dim=self.embed_dim[i+1] if i<(self.num_depth-1) else None,
                                        downsample_flag=True if i<(self.num_depth-1) else False,attn_type=attn_type)


            self.stages.append(single_stage)

        self.final_embed_dim = self.embed_dim[-1]
        self.norm = norm_layer(self.final_embed_dim)

        self.head = nn.Linear(self.final_embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        print('n_window')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        # trunc_normal_(self.absolute_pos_embed, std=.02)
            
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.final_embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self,x):
        x = self.patch_embed(x)      

        # B,H,W,D = x.shape
        # x = x.view(B,-1,D).contiguous()
        # x = x + self.absolute_pos_embed
        # x = x.view(B,H,W,D).contiguous()
     

        for i,stage in enumerate(self.stages):
            x = stage(x)  

        x = self.norm(x)
        x = x.flatten(1,2).contiguous().mean(dim=1).squeeze()
        x = self.head(x)
        return x



@register_model
def angular_tiny_exp_224(pretrained=False, **kwargs):
    model = AngularViT(embed_dim=[64,128,256,512], depth=[2,4,18,2], num_heads=[1,2,4,8], 
                    n_window=[(8,7),(4,3),(2,1),(1,1)],
                    kv_pooling=[0,0,0,0],  mlp_ratio=4., attn_type='exp', **kwargs)
    return model


@register_model
def angular_tiny_quad_224(pretrained=False, **kwargs):
    model = AngularViT(embed_dim=[64,128,256,512], depth=[2,4,18,2], num_heads=[1,2,4,8], 
                    n_window=[(8,7),(4,3),(2,1),(1,1)],
                    kv_pooling=[0,0,0,0],  mlp_ratio=4., attn_type='quad', **kwargs)
    return model


@register_model
def angular_small_quad_224(pretrained=False, **kwargs):
    model = AngularViT(embed_dim=[80,160,320,640], depth=[3,6,21,3], num_heads=[1,2,4,8], 
                    n_window=[(8,7),(4,3),(2,1),(1,1)],
                    kv_pooling=[0,0,0,0],  mlp_ratio=4., attn_type='quad', **kwargs)
    return model

@register_model
def angular_base_quad_224(pretrained=False, **kwargs):
    model = AngularViT(embed_dim=[96,192,384,768], depth=[4,8,24,4], num_heads=[2,4,8,16], 
                    n_window=[(8,7),(4,3),(2,1),(1,1)],
                    kv_pooling=[0,0,0,0],  mlp_ratio=4., attn_type='quad', **kwargs)
    return model



@register_model
def angular_tiny_cos_224(pretrained=False, **kwargs):
    model = AngularViT(embed_dim=[64,128,256,512], depth=[2,4,18,2], num_heads=[1,2,4,8], 
                    n_window=[(8,7),(4,3),(2,1),(1,1)],
                    kv_pooling=[0,0,0,0],  mlp_ratio=4., attn_type='cos', **kwargs)
    return model


@register_model
def angular_small_cos_224(pretrained=False, **kwargs):
    model = AngularViT(embed_dim=[80,160,320,640], depth=[3,6,21,3], num_heads=[1,2,4,8], 
                    n_window=[(8,7),(4,3),(2,1),(1,1)],
                    kv_pooling=[0,0,0,0],  mlp_ratio=4., attn_type='cos', **kwargs)
    return model

@register_model
def angular_base_cos_224(pretrained=False, **kwargs):
    model = AngularViT(embed_dim=[96,192,384,768], depth=[4,8,24,4], num_heads=[2,4,8,16], 
                    n_window=[(8,7),(4,3),(2,1),(1,1)],
                    kv_pooling=[0,0,0,0],  mlp_ratio=4., attn_type='cos', **kwargs)
    return model


if __name__=='__main__':


    # a = torch.tensor([-6])
    # b = torch.exp(a)
    # b = b.half()
    # print(b)
    # exit()
    x = torch.rand(2,3,224,224).cuda()

    model = create_model('angular_tiny_exp_224')
    # print(timm.list_models())
    # exit()
    # model = create_model('vit_small_patch16_224')
    model = model.cuda()


    # cs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
    #                                        print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

  
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(n_parameters)
    with torch.no_grad():
        x= model(x)
        print(x.size())

    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            