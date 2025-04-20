import os
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import argparse
import copy
from omegaconf import OmegaConf
from einops import rearrange
from pytorch_lightning import seed_everything
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler

from ldm.models.diffusion.ddim import DDIMSampler

import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import pickle

'''
python run_styleid_class.py 
        '''
class StyleID:
    def __init__(
        self,
        model_config_path="models/ldm/stable-diffusion-v1/v1-inference.yaml",
        model_ckpt_path="/Users/ran/Documents/Datasets/HE_Patch/sd-v1-4.ckpt",
        precomputed_dir="precomputed_feats",
        output_dir="output",
        device="cpu",
        seed=22,
        gamma=0.75,
        temperature=1.5,
        attn_layers="6,7,8,9,10,11",
        precision="full",
    ):
        self.model_config_path = model_config_path
        self.model_ckpt_path = model_ckpt_path
        self.precomputed_dir = precomputed_dir
        self.output_dir = output_dir

        # 创建输出喝预计算特征目录
        os.makedirs(output_dir, exist_ok=True)
        if precomputed_dir:
            os.makedirs(precomputed_dir, exist_ok=True)
            
        # if device is None:
        #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # else:   
        #     self.device = torch.device(device)
        if device is None:
            if force_cpu:
                self.device = torch.device("cpu")
                print("强制使用 CPU")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("使用 CUDA 加速")
            elif hasattr(torch, 'has_mps') and torch.has_mps and torch.backends.mps.is_available():
                # Mac 上的 MPS 设备会导致问题，所以我们强制使用 CPU
                print("检测到 MPS 设备，但因兼容性问题强制使用 CPU")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
                print("使用 CPU")
        else:
            if device == "mps" or (isinstance(device, torch.device) and device.type == "mps"):
                print("MPS 设备可能导致兼容性问题，强制使用 CPU")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)
            
        # 在 CPU 上设置精度为 "full"
        if self.device.type == "cpu":
            self.precision = "full"
            print("在 CPU 上使用 full 精度")

        seed_everything(seed)

        self.seed = seed
        self.gamma = gamma
        self.temperature = temperature
        self.attn_layers = list(map(int, attn_layers.split(',')))
        self.precision = precision

        self.feat_maps = []
        # 加载模型和初始化采样器 
        print(f"正在加载模型: {model_ckpt_path}")
        self.model = self._load_model(model_ckpt_path, model_config_path)
        
        # 初始化 DDIM 采样器
        print("正在初始化 DDIM 采样器")
        self.sampler = DDIMSampler(self.model)
        
        # 获取 UNet 模型以便提取特征
        self.unet = self.model.model.diffusion_model
        
        print("模型加载完成")

    def _load_model(self, model_path, config_path):
        print(f"Loading model from {model_path} with config {config_path}")
        config = OmegaConf.load(config_path)
        model = instantiate_from_config(config.model).to(self.device)

        # 加载权重
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state_dict, strict=False)

        # 设置为评估模式 ✅ 关键一步
        # eval重要，是pytorch中用于将模型设置为评估模式的方法
        '''
        ✨ 1. Dropout 层
        在训练时：Dropout 会随机丢弃部分神经元（提高模型泛化能力）
        在评估时（.eval()）：Dropout 被关闭（不会丢弃神经元）
        📊 2. BatchNorm 层
        在训练时：使用当前 batch 的均值和方差
        在评估时：使用训练时保存的全局均值和方差
                '''
        model.eval()

        return model
    
    def _setup_ddim_sampler(self, ddim_steps, ddim_eta=0.0):
        """设置 DDIM 采样器参数"""
        self.sampler.make_schedule(
            ddim_num_steps=ddim_steps, 
            ddim_eta=ddim_eta, 
            verbose=False
        )
        
        # 获取时间步长
        self.time_range = np.flip(self.sampler.ddim_timesteps)
        
        # 创建时间步和索引的映射
        self.idx_time_dict = {}
        self.time_idx_dict = {}
        for i, t in enumerate(self.time_range):
            self.idx_time_dict[t] = i
            self.time_idx_dict[i] = t

    def _save_feature_map(self, feature_map, filename, time):
        # 保存单个特征图
        # cur_idx = self.idx_time_dict[time]
        # self.feat_maps[cur_idx] = feat_maps
        cur_idx = self.idx_time_dict[time]
        self.feat_maps[cur_idx][f"{filename}"] = feature_map

    def _save_feature_maps(self, blocks, i, feature_type="input_block"):
        # 保存特定块的
        block_idx = 0
        for block_idx, block in enumerate(blocks):
            if block_idx > 1 and "SpatialTransformer" in str(type(block[1])):
                if  block_idx in self.attn_layers:
                    q = block[1].transformer_blocks[0].attn1.q
                    k = block[1].transformer_blocks[0].attn1.k
                    v = block[1].transformer_blocks[0].attn1.v
                    self._save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", i)
                    self._save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", i)
                    self._save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", i)
            block_idx += 1


    # 回掉函数事先定义好，但不会立刻执行的，等到未来某个时间触发的函数
    # 外层 callback → 调用 save_feature_maps → 多次调用 save_feature_map
    # 保存中间特征图，可视化，查看图像是怎么一步步生成的
    def _save_feature_maps_callback(self, i):
        """保存特征图的回调函数"""
        # 原代码调用了不存在的方法 _save_feat_map
        # 修改为调用已有的 _save_feature_maps 方法
        self._save_feature_maps(self.unet.output_blocks, i, "output_block")

    def _ddim_sampler_callback(self, pred_x0, xt, i):
        """DDIM 采样器回调，用于保存特征图"""
        self._save_feature_maps_callback(i)
        self._save_feature_map(xt, 'z_enc', i)

    def _feat_merge(self, cnt_feat, sty_feat, start_step=0):
    # 计算权重
        result_feat_maps = [{'config':{'gamma':self.gamma, 'T':self.temperature, 'timestep':_,}} for _ in range(50)]

        for i in range(len(result_feat_maps)):
            if i < (50 - start_step):
                continue
            # 修复变量使用错误 - 原代码在循环内重复使用 cnt_feat 和 sty_feat
            cnt_feat_i = cnt_feat[i]
            sty_feat_i = sty_feat[i]
            ori_keys = sty_feat_i.keys()

            for ori_key in ori_keys:
                if ori_key[-1] == 'q':
                    result_feat_maps[i][ori_key] = cnt_feat_i[ori_key]
                if ori_key[-1] == 'k' or ori_key[-1] == 'v':
                    result_feat_maps[i][ori_key] = sty_feat_i[ori_key]
        return result_feat_maps

    def _load_img(self, path, h=512, w=512):
        """加载并图像"""
        image = Image.open(path).convert("RGB")
        x, y = image.size
        print(f"Loaded input image of size ({x}, {y}) from {path}")

        image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.*image - 1.
    
    def _adain(self, cnt_feat, sty_feat):
        """自适应实例归一化
        把 cnt_feat 的特征 标准化 后，重新赋予 sty_feat 的均值和方差，最终返回一个新的 feature map，风格来自 style 图，结构来自 content 图。
        """
        cnt_mean = cnt_feat.mean(dim=[0, 2, 3], keepdim=True)
        cnt_std = cnt_feat.std(dim=[0, 2, 3], keepdim=True)
        sty_mean = sty_feat.mean(dim=[0, 2, 3], keepdim=True)
        sty_std = sty_feat.std(dim=[0, 2, 3], keepdim=True)
        # 取 每个通道（dim=0, 2, 3 → batch, height, width）上的 mean 和 std
        output = ((cnt_feat-cnt_mean)/cnt_std)*sty_std + sty_mean
        # 内容图特征标准化（这一步相当于去掉内容图的本身特征，用风格图的统计信息重建分布）
        # 核心意义：保留内容图的形状，但让它的特征分布看起来像风格图

        return output  
    
    def transfer_style(
            self,
            content_img_path,
            style_img_path,
            output_name = None,
            ddim_steps=50,
            ddim_eta=0.0,
            ddim_scale=1.0,
            start_step=49,
            use_adain=True,
            use_attn_injection=True,
            h=512,
            w=512,
    ):
        self._setup_ddim_sampler(ddim_steps, ddim_eta)

        if output_name is None:
            output_name = os.path.basename(content_img_path).split('.')[0]
            style_name = os.path.basename(style_img_path).split('.')[0]
            output_name = f"{output_name}_stylized_{style_name}.png"

        output_path = os.path.join(self.output_dir, output_name)

        self.feat_maps = [{
            'config':{
                'gamma':self.gamma,
                'T':self.temperature,
            }
        }for _ in range(50)]

        uc = self.model.get_learned_conditioning([""])

        shape = (4, h // 8, w // 8)   # latent space中的大小



        # 加载风格图像并获取特征
        print(f"Loading style image from {style_img_path}")
        sty_feat_name = os.path.join(
            self.precomputed_dir,
            os.path.basename(style_img_path).split('.')[0] + "_sty.pkl"
        )
        
        if self.precomputed_dir and os.path.exists(sty_feat_name):
            with open(sty_feat_name, "rb") as h:
                sty_feat = pickle.load(h)
                sty_z_enc = torch.clone(sty_feat[0]['z_enc'])
        else:
            init_sty = self._load_img(style_img_path, 512, 512).to(self.device)
            init_sty = self.model.get_first_stage_encoding(
                self.model.encode_first_stage(init_sty) # 送入vae编码器，得到编码后表示feature map
            )
            sty_z_enc, _ = self.sampler.encode_ddim(
                init_sty.clone(),
                num_steps=ddim_steps,
                unconditional_conditioning = uc,
                end_step = self.time_idx_dict[ddim_steps-1-start_step],
                callback_ddim_timesteps = ddim_steps,
                img_callback = self._ddim_sampler_callback,
            )
            sty_feat = copy.deepcopy(self.feat_maps)
            sty_z_enc = self.feat_maps[0]['z_enc']
            
            # 保存风格特征
            if self.precomputed_dir:
                with open(sty_feat_name, 'wb') as h:
                    pickle.dump(sty_feat, h)
        
        # 加载内容图像并获取特征
        print(f"处理内容图像: {content_img_path}")
        cnt_feat_name = os.path.join(
            self.precomputed_dir, 
            os.path.basename(content_img_path).split('.')[0] + '_cnt.pkl'
        )

                # 检查是否存在预计算的内容特征
        if self.precomputed_dir and os.path.isfile(cnt_feat_name):
            print(f"加载预计算内容特征: {cnt_feat_name}")
            with open(cnt_feat_name, 'rb') as h:
                cnt_feat = pickle.load(h)
                cnt_z_enc = torch.clone(cnt_feat[0]['z_enc'])
        else:
            # 加载并编码内容图像
            init_cnt = self._load_img(content_img_path, 512, 512).to(self.device)
            init_cnt = self.model.get_first_stage_encoding(
                self.model.encode_first_stage(init_cnt)
            )
            
            # 内容图像 DDIM 反转
            cnt_z_enc, _ = self.sampler.encode_ddim(
                init_cnt.clone(), 
                num_steps=ddim_steps, 
                unconditional_conditioning=uc,
                end_step=self.time_idx_dict[ddim_steps-1-start_step],
                callback_ddim_timesteps=ddim_steps,
                img_callback=self._ddim_sampler_callback
            )
            cnt_feat = copy.deepcopy(self.feat_maps)
            cnt_z_enc = self.feat_maps[0]['z_enc']
            
            # 保存内容特征
            # if self.precomputed_dir:
            #     with open(cnt_feat_name, 'wb') as h:
            #         pickle.dump(cnt_feat, h)
        
        # if self.precision == "autocast" and torch.cuda.is_available():
        #     precision_scope = lambda: autocast("cuda")
        # else:
        precision_scope = nullcontext
        
        with torch.no_grad():
            with precision_scope():
                with self.model.ema_scope():
                    # AdaIN 处理潜在编码
                    if not use_adain:
                        adain_z_enc = cnt_z_enc
                    else:
                        adain_z_enc = self._adain(cnt_z_enc, sty_z_enc)
                    
                    # 合并特征
                    # merged_feat_maps = self._feat_merge(
                    #     cnt_feat, sty_feat, start_step=start_step
                    # )
                    self.feat_maps = self._feat_merge(cnt_feat, sty_feat, start_step=start_step)
                    
                    # 如果不使用注意力注入则设置为 None
                    if not use_attn_injection:
                        self.feat_maps = None
                    
                    # 采样生成结果
                    samples_ddim, _ = self.sampler.sample(
                        S=ddim_steps,
                        batch_size=1,
                        shape=shape,
                        verbose=False,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        x_T=adain_z_enc,
                        injected_features=self.feat_maps,
                        start_step=start_step,
                        
                    )
                    
     # 解码生成图像
                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp(
                        (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                    )
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                    x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    
                    # 保存结果图像
                    img.save(output_path)

        print(f"风格迁移完成，结果保存至: {output_path}")
        return output_path
    
    def batch_transfer(
        self, 
        content_dir, 
        style_dir, 
        output_dir=None, 
        **kwargs
    ):
        """
        批量处理目录中的图像
        
        参数:
            content_dir: 内容图片目录
            style_dir: 风格图片目录
            output_dir: 输出目录，默认为实例化时设置的目录
            **kwargs: 传递给 transfer_style 的参数
            
        返回:
            输出文件路径列表
        """
        if output_dir:
            old_output_dir = self.output_dir
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有内容和风格图片
        content_imgs = [f for f in os.listdir(content_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        style_imgs = [f for f in os.listdir(style_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 排序以保持一致性
        content_imgs.sort()
        style_imgs.sort()
        results = []
        total = len(content_imgs) * len(style_imgs)
        
        # 批量处理
        with tqdm(total=total, desc="批处理风格迁移") as pbar:
            print("1")
            for content_img in content_imgs:
                for style_img in style_imgs:
                    content_path = os.path.join(content_dir, content_img)
                    style_path = os.path.join(style_dir, style_img)
                    print("2")
                    
                    result_path = self.transfer_style(
                        content_path, style_path, **kwargs
                    )
                    results.append(result_path)
                    print(f"3")
                    # except Exception as e:
                    #     print(f"4")
                    #     print(f"处理 {content_img} 和 {style_img} 时出错: {e}")
                    
                    pbar.update(1)
        print("5")
        # 恢复原始输出目录
        if output_dir:
            self.output_dir = old_output_dir
            
        return results
    

# 命令行接口
def main():
    parser = argparse.ArgumentParser(description="StyleID 风格迁移")
    parser.add_argument('--cnt', type=str, default='./data/cnt', help='内容图像目录或路径')
    parser.add_argument('--sty', type=str, default='./data/sty', help='风格图像目录或路径')
    parser.add_argument('--output_path', type=str, default='output', help='输出目录')
    parser.add_argument('--precomputed', type=str, default='./precomputed_feats', help='预计算特征存储目录')
    parser.add_argument('--model_config', type=str, default='models/ldm/stable-diffusion-v1/v1-inference.yaml', help='模型配置')
    parser.add_argument('--ckpt', type=str, default='/Users/ran/Documents/Datasets/HE_Patch/sd-v1-4.ckpt', help='模型权重')
    parser.add_argument('--ddim_steps', type=int, default=50, help='DDIM 步数')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--start_step', type=int, default=49, help='开始步骤')
    parser.add_argument('--gamma', type=float, default=0.75, help='查询保留超参数')
    parser.add_argument('--T', type=float, default=1.5, help='注意力温度超参数')
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11', help='注入特征层')
    parser.add_argument('--precision', type=str, default='full', help='精度: full 或 autocast')
    parser.add_argument("--without_init_adain", action='store_true', help='禁用 AdaIN')
    parser.add_argument("--without_attn_injection", action='store_true', help='禁用注意力注入')
    parser.add_argument('--seed', type=int, default=22, help='随机种子')
    
    opt = parser.parse_args()
    
    # 创建 StyleID 模型
    model = StyleID(
        model_config_path=opt.model_config,
        model_ckpt_path=opt.ckpt,
        precomputed_dir=opt.precomputed,
        output_dir=opt.output_path,
        seed=opt.seed,
        gamma=opt.gamma,
        temperature=opt.T,
        attn_layers=opt.attn_layer,
        precision=opt.precision
    )
 # 检查输入是文件还是目录
    if os.path.isdir(opt.cnt) and os.path.isdir(opt.sty):
        # 批量处理
        print(f"批量处理内容目录: {opt.cnt} 和风格目录: {opt.sty}")
        model.batch_transfer(
            content_dir=opt.cnt,
            style_dir=opt.sty,
            ddim_steps=opt.ddim_steps,
            ddim_eta=opt.ddim_eta,
            start_step=opt.start_step,
            use_adain=not opt.without_init_adain,
            use_attn_injection=not opt.without_attn_injection
        )
    else:
        # 单图处理
        print(f"处理单张图像: {opt.cnt} 和 {opt.sty}")
        model.transfer_style(
            content_img_path=opt.cnt,
            style_img_path=opt.sty,
            ddim_steps=opt.ddim_steps,
            ddim_eta=opt.ddim_eta,
            start_step=opt.start_step,
            use_adain=not opt.without_init_adain,
            use_attn_injection=not opt.without_attn_injection
        )
    
    
if __name__ == "__main__":
    main()