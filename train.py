import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from tqdm import tqdm
import pandas as pd
import numpy as np

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer

import random
import traceback
import sys

from utils import Train_Report
# ----------------------------------------
from model.center_loss import CenterLoss
from model.dit import DiT
from model.flow_matcher import FlowMatcher


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


class Trainer:
    def __init__(self, args, data_loader):
        self.args = args
        self.train_data_loader = data_loader['train']
        self.val_data_loader = data_loader['val']
        self.test_data_loader = data_loader['test']
        self.gzsl_loader = data_loader['gzsl']
        # Memory bank
        self.memory_bank = {}  # 结构: {class_idx: [(score, feature_tensor), ...]}
        self.bank_min_scores = {}  # 结构: {class_idx: min_score} 缓存每类当前的最低分，加速判断
        self.bank_initialized = False  # 标记是否已经初始化了 key
        self.bank_max_p = getattr(self.args, 'max_p', 20)  # 每个类别存多少个样本
        # Accelerator
        self.accelerator_project_config = ProjectConfiguration(project_dir=args.work_dir)
        self.accelerator = Accelerator(mixed_precision=args.mixed_precision,
                                       project_config=self.accelerator_project_config)
        if self.accelerator.is_main_process:
            if args.work_dir is not None:
                os.makedirs(args.work_dir, exist_ok=True)

        # Weight data type
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Model
        self.load_checkpoint()
        self.text_prompt()

        # +++++ [Flow Matching] +++++
        self.flow_matcher = FlowMatcher()

        self.dit = DiT(in_channels=self.args.in_channels, hidden_size=self.args.hidden_size, depth=self.args.depth,
                       num_heads=self.args.num_heads)

        self.unseen_num = self.args.unseen_label
        self.unseen_labels = np.load(self.args.unseen_label_path)

        # 确定总类别数并区分 Seen/Unseen
        if 'ntu60' in self.args.unseen_label_path:
            self.total_classes = np.arange(60)
        elif 'pku51' in self.args.unseen_label_path:
            self.total_classes = np.arange(51)
        else:  # ntu120
            self.total_classes = np.arange(120)

        self.seen_labels = np.where(~np.isin(self.total_classes, self.unseen_labels))[0]
        self.seen_num = len(self.seen_labels)

        self.seen_label_mapping = {label: idx for idx, label in enumerate(self.seen_labels)}
        self.unseen_label_mapping = {label: idx for idx, label in enumerate(self.unseen_labels)}

        self.center_loss_fn = CenterLoss(num_classes=len(self.total_classes), feat_dim=self.args.in_channels,
                                         use_gpu=True)  # 假设 in_channels 是特征维度

        # Optimizer
        params_to_opt = list(self.dit.parameters()) + list(self.center_loss_fn.parameters())
        self.optimizer = torch.optim.AdamW(params_to_opt, lr=self.args.learning_rate,
                                           weight_decay=self.args.weight_decay)
        # 将 center_loss_fn 移至 device
        self.center_loss_fn.to(self.accelerator.device)
        # Scheduler
        self.lr_scheduler = get_scheduler(self.args.lr_scheduler, optimizer=self.optimizer,
                                          num_warmup_steps=self.args.num_warmup, num_training_steps=self.args.num_iter)

        self.dit, self.optimizer, self.lr_scheduler, self.train_data_loader, self.val_data_loader, self.test_data_loader = self.accelerator.prepare(
            self.dit, self.optimizer, self.lr_scheduler, self.train_data_loader, self.val_data_loader,
            self.test_data_loader)

    def load_checkpoint(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path,
                                                          subfolder="text_encoder")
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.requires_grad_(False)
    def text_prompt(self):
        text_embed_csv = []
        df = pd.read_csv('./data/class_lists/ntu120.csv')  # 'ntu120.csv' includes 'ntu60.csv'
        prompts = df['label'].values.tolist()
        for prompt in prompts:
            text_inputs = self.tokenizer(prompt, padding="max_length", max_length=35, truncation=True,
                                         return_tensors="pt").to(self.accelerator.device)
            embeds = self.text_encoder(text_inputs.input_ids)
            prompt_embeds = embeds['last_hidden_state']
            pooled_prompt_embeds = embeds['pooler_output']
            text_embed_csv.append(
                torch.cat((prompt_embeds, pooled_prompt_embeds.unsqueeze(1)), dim=1).detach().clone().to(
                    self.accelerator.device))

        text_embed_txt = []
        with open('./data/class_lists/ntu120_llm.txt', 'r',
                  encoding='utf-8') as file:
            content_list = file.readlines()
        prompts = [line.strip() for line in content_list]
        for prompt in prompts:
            text_inputs = self.tokenizer(prompt, padding="max_length", max_length=35, truncation=True,
                                         return_tensors="pt").to(self.accelerator.device)
            embeds = self.text_encoder(text_inputs.input_ids)
            prompt_embeds = embeds['last_hidden_state']
            pooled_prompt_embeds = embeds['pooler_output']
            text_embed_txt.append(
                torch.cat((prompt_embeds, pooled_prompt_embeds.unsqueeze(1)), dim=1).detach().clone().to(
                    self.accelerator.device))

        self.text_embed = torch.cat((torch.cat(text_embed_csv, dim=0), torch.cat(text_embed_txt, dim=0)), dim=-1)
    # ntu数据集
    # def text_prompt(self):
    #     """ 
    #     - Holistic (txt): 提取自 ntu120_holistic.txt -> [120, 36, 1024]
    #     - Phase (json): 提取自 phase_des.json (3句拼接) -> [120, 36, 1024]
    #     - 最终输出: self.text_embed [120, 36, 2048] (dim=-1 拼接)
    #     """
    #     import os
    #     import json
    #     import string
    #     import torch
    #     import pandas as pd
    #     import warnings

    #     device = self.accelerator.device
    #     dtype = getattr(self, "weight_dtype", torch.float16)
    #     TE = self.text_encoder
    #     TK = self.tokenizer
    #     TE.eval()

    #     # 辅助：规范化用于匹配的 key
    #     def normalize_key(s: str) -> str:
    #         return "".join(e for e in s.lower() if e.isalnum())

    #     # 1) 获取顺序基准 (Labels)
    #     labels_df = pd.read_csv('data/class_lists/ntu120_labels.csv')
    #     labels = labels_df['label'].values.tolist()

    #     # ----------------------------------------------------------------------
    #     # 2) 处理 ntu120_holistic.txt (Holistic 路径)
    #     # ----------------------------------------------------------------------
    #     with open('data/class_lists/ntu120_holistic.txt', 'r', encoding='utf-8') as f:
    #         hol_lines = [ln.strip() for ln in f if ln.strip()]
    #     assert len(hol_lines) == len(labels), "Holistic 文件行数与 Labels 不一致"

    #     text_embed_holistic = []
    #     with torch.no_grad():
    #         for prompt in hol_lines:
    #             text_inputs = TK(prompt, padding="max_length", max_length=35, truncation=True,
    #                              return_tensors="pt").to(device)
    #             embeds = TE(text_inputs.input_ids)
                
    #             prompt_embeds = embeds['last_hidden_state'] # [1, 35, 1024]
    #             pooled_embeds = embeds['pooler_output'].unsqueeze(1) # [1, 1, 1024]
                
    #             # 拼接为 [1, 36, 1024]
    #             combined = torch.cat((prompt_embeds, pooled_embeds), dim=1)
    #             text_embed_holistic.append(combined.detach().clone())

    #     holistic_final = torch.cat(text_embed_holistic, dim=0) # [120, 36, 1024]

    #     # ----------------------------------------------------------------------
    #     # 3) 处理 phase_des.json (Phased 路径)
    #     # ----------------------------------------------------------------------
    #     phase_path = 'data/class_lists/phase_des.json'
    #     with open(phase_path, 'r', encoding='utf-8') as f:
    #         phase_raw = json.load(f)

    #     # 建立匹配映射
    #     phase_map_norm = {normalize_key(k): (v['phase'] if isinstance(v, dict) else v) for k, v in phase_raw.items()}

    #     text_embed_phase = []
    #     with torch.no_grad():
    #         for lbl in labels:
    #             vv = phase_map_norm.get(normalize_key(lbl), None)
    #             if not (isinstance(vv, list) and len(vv) == 3):
    #                 warnings.warn(f"Phase 匹配失败，兜底处理: {lbl}")
    #                 vv = [lbl, lbl, lbl]
                
    #             # 拼接三句话逻辑
    #             joined_phase = f"{vv[0]} ; {vv[1]} ; {vv[2]}"
    #             text_inputs = TK(joined_phase, padding="max_length", max_length=35, truncation=True,
    #                              return_tensors="pt").to(device)
    #             embeds = TE(text_inputs.input_ids)
                
    #             prompt_embeds = embeds['last_hidden_state'] # [1, 35, 1024]
    #             pooled_embeds = embeds['pooler_output'].unsqueeze(1) # [1, 1, 1024]
                
    #             # 拼接为 [1, 36, 1024]
    #             combined = torch.cat((prompt_embeds, pooled_embeds), dim=1)
    #             text_embed_phase.append(combined.detach().clone())

    #     phase_final = torch.cat(text_embed_phase, dim=0) # [120, 36, 1024]

    #     self.text_embed = torch.cat((holistic_final, phase_final), dim=-1).to(device, dtype=dtype)

    # pku51数据集
    # def text_prompt(self):
    #     text_embed_csv = []
    #     df = pd.read_csv('./data/class_lists/pku51.csv')  # 'ntu120.csv' includes 'ntu60.csv'
    #     prompts = df['label'].values.tolist()
    #     for prompt in prompts:
    #         text_inputs = self.tokenizer(prompt, padding="max_length", max_length=35, truncation=True,
    #                                      return_tensors="pt").to(self.accelerator.device)
    #         embeds = self.text_encoder(text_inputs.input_ids)
    #         prompt_embeds = embeds['last_hidden_state']
    #         pooled_prompt_embeds = embeds['pooler_output']
    #         text_embed_csv.append(
    #             torch.cat((prompt_embeds, pooled_prompt_embeds.unsqueeze(1)), dim=1).detach().clone().to(
    #                 self.accelerator.device))
    
    #     text_embed_txt = []
    #     with open('./data/class_lists/pku51_llm.txt', 'r',
    #               encoding='utf-8') as file:
    #         content_list = file.readlines()
    #     prompts = [line.strip() for line in content_list]
    #     for prompt in prompts:
    #         text_inputs = self.tokenizer(prompt, padding="max_length", max_length=35, truncation=True,
    #                                      return_tensors="pt").to(self.accelerator.device)
    #         embeds = self.text_encoder(text_inputs.input_ids)
    #         prompt_embeds = embeds['last_hidden_state']
    #         pooled_prompt_embeds = embeds['pooler_output']
    #         text_embed_txt.append(
    #             torch.cat((prompt_embeds, pooled_prompt_embeds.unsqueeze(1)), dim=1).detach().clone().to(
    #                 self.accelerator.device))
    
    #     self.text_embed = torch.cat((torch.cat(text_embed_csv, dim=0), torch.cat(text_embed_txt, dim=0)), dim=-1)

    def save_best_model(self):
        save_path = os.path.join(self.args.work_dir, 'best')
        self.accelerator.save_state(save_path)

    def train(self, train_log, global_step):
        self.dit.train()
        self.dit.requires_grad_(True)
        report = Train_Report()
        start = time.time()

        for idx, (features, labels) in tqdm(enumerate(self.train_data_loader)):
            with self.accelerator.accumulate(self.dit):
                # 1. 准备数据 x1 (真实骨架特征)
                x1 = features.to(self.accelerator.device, dtype=self.weight_dtype).unsqueeze(dim=1)

                # 2. 准备 x0 (随机噪声)
                x0 = torch.randn_like(x1)

                # 3. 准备 t (时间)
                t = self.flow_matcher.sample_t(shape=(self.args.batch_size,), device=x1.device)

                # 4. 计算 Flow Matching 路径和目标
                x_t, u_t = self.flow_matcher.get_train_tuple(x0, x1, t)

                # 5. 准备文本嵌入 (c_pos 和 c_neg)
                with torch.no_grad():
                    # 5.1 正样本
                    text_embed_pos = self.text_embed[labels.long()].to(self.accelerator.device, dtype=self.weight_dtype)

                    # 5.2 负样本
                    labels_r = torch.Tensor(random.choices(self.seen_labels, k=self.args.batch_size)).to(
                        self.accelerator.device)
                    text_embed_neg = self.text_embed[labels_r.long()].to(self.accelerator.device,
                                                                         dtype=self.weight_dtype)

                    # 5.3 掩码 (mask)
                    mask = (labels != labels_r).to(self.accelerator.device, dtype=self.weight_dtype)

                # 6. 准备模型输入 (合并 pos 和 neg)
                x_t_batch = x_t.repeat(2, 1, 1)
                t_batch = t.repeat(2)
                text_embed_batch = torch.cat((text_embed_pos, text_embed_neg), dim=0)
                u_t_target = u_t.repeat(2, 1, 1)

                # 7. 模型前向传播
                v_pred_batch = self.dit(x_t_batch, t_batch,
                                        text_embed_batch[:, -1, :],
                                        text_embed_batch[:, :-1, :])

                # 8. 计算损失
                loss_fm = F.mse_loss(v_pred_batch[:self.args.batch_size],
                                     u_t_target[:self.args.batch_size]) * self.args.fm_weight

                pos_error = F.mse_loss(v_pred_batch[:self.args.batch_size], u_t_target[:self.args.batch_size],
                                       reduce=False).mean(dim=(1, 2))
                neg_error = F.mse_loss(v_pred_batch[self.args.batch_size:], u_t_target[self.args.batch_size:],
                                       reduce=False).mean(dim=(1, 2))

                triplet = torch.clamp(pos_error - neg_error + self.args.margin, min=0.0) * mask
                loss_triplet = triplet.mean() * self.args.t_weight

                # +++++ [Center Loss 计算] +++++
                # 核心逻辑：从 x_t, t, v_pred 反推预测的 x1 (Predicted Clean Data)
                # 公式: x_1_pred = x_t + (1 - t) * v_pred
                # 注意维度匹配: t 需要 reshape 为 (B, 1, 1)
                v_pred_pos = v_pred_batch[:self.args.batch_size]  # (B, 1, C)
                t_reshaped = t.view(self.args.batch_size, 1, 1)

                # 只对正样本计算 Center Loss (Feature Alignment)
                x_t_pos = x_t_batch[:self.args.batch_size]

                # 重构预测的特征
                pred_x1 = x_t_pos + (1 - t_reshaped) * v_pred_pos

                # 展平特征以输入 CenterLoss (B, C)
                pred_x1_flat = pred_x1.view(self.args.batch_size, -1)

                # 计算损失

                # loss_center = self.center_loss_fn(pred_x1_flat, labels.long()) * 0.002
                if global_step < 2000:
                    loss_center = self.center_loss_fn(pred_x1_flat.detach(), labels.long()) * self.args.c_weight
                else:
                    loss_center = self.center_loss_fn(pred_x1_flat, labels.long()) * self.args.c_weight

                loss = loss_fm + loss_triplet + loss_center

                # 9. 反向传播
                reduced_loss = self.accelerator.gather(loss).mean()
                reduced_loss_fm = self.accelerator.gather(loss_fm).mean()
                reduced_loss_triplet = self.accelerator.gather(loss_triplet).mean()
                reduced_loss_center = self.accelerator.gather(loss_center).mean()
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                if self.accelerator.is_main_process:
                    report.update(self.args.batch_size, reduced_loss.item(),
                                  reduced_loss_fm.item(), reduced_loss_triplet.item(), reduced_loss_center.item())

            global_step += 1

            if global_step % self.args.log_iter == 0 or idx == len(self.train_data_loader) - 1:
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                period_time = time.time() - start
                prefix_str = f'Iter[{global_step}/{self.args.num_iter}]\t'
                result_str = report.result_str_cl(lr, period_time)
                train_log.write(prefix_str + result_str)
                start = time.time()
                report.__init__()

            if global_step % self.args.save_iter == 0:
                save_path = os.path.join(self.args.work_dir, f'checkpoint-{global_step}')
                self.accelerator.save_state(save_path)

        return global_step

    def init_memory_bank(self, num_classes, max_p=50):
        """
        初始化全局 Prototype Memory Bank
        Structure: {class_idx: [(score, feature_tensor), ...]}
        """
        self.bank_max_p = max_p
        self.memory_bank = {c: [] for c in range(num_classes)}
        # 用于记录该类当前的最小置信度，避免频繁排序
        self.bank_min_scores = {c: -1.0 for c in range(num_classes)}

    def update_memory_bank(self, class_level_sim, real_features):
        """
        根据当前 Batch 的预测结果更新 Memory Bank (Streaming Update)
        class_level_sim: [B, C]
        real_features: [B, Dim]
        """
        # 1. 获取当前 Batch 的伪标签和置信度
        # 使用 Softmax 后的 Max Prob 作为置信度
        probs = torch.softmax(class_level_sim * 10.0, dim=-1)
        confidences, pseudo_labels = torch.max(probs, dim=-1)

        # 2. 按类别批量处理，减少循环次数
        unique_labels = torch.unique(pseudo_labels)

        for c in unique_labels:
            c = c.item()
            # 找到当前 Batch 中预测为 c 的样本
            idxs = (pseudo_labels == c).nonzero(as_tuple=True)[0]

            c_confs = confidences[idxs]
            c_feats = real_features[idxs]

            # 遍历该类别的候选样本
            for i in range(len(idxs)):
                score = c_confs[i].item()
                feature = c_feats[i].detach()  # Detach to save memory if needed

                current_bank = self.memory_bank[c]

                # 情况 A: Bank 未满 -> 直接添加
                if len(current_bank) < self.bank_max_p:
                    current_bank.append((score, feature))
                    # 如果填满或刚添加，重新排序以维护最小分数值
                    current_bank.sort(key=lambda x: x[0])
                    self.bank_min_scores[c] = current_bank[0][0]

                # 情况 B: Bank 已满 -> 比较置信度
                else:
                    # 如果当前样本分数 > Bank 中的最小值
                    if score > self.bank_min_scores[c]:
                        # 移除分数最小的 (因为已经排好序，第一个就是最小)
                        current_bank.pop(0)
                        # 添加新的
                        current_bank.append((score, feature))
                        # 重新排序
                        current_bank.sort(key=lambda x: x[0])
                        # 更新最小值
                        self.bank_min_scores[c] = current_bank[0][0]

    def get_prototypes_from_bank(self, num_classes, feature_dim):
        """
        从 Memory Bank 中计算每个类别的 Prototype
        """
        prototypes = torch.zeros(
            num_classes,
            feature_dim,
            device=self.accelerator.device,
            dtype=self.weight_dtype
        )
        for c in range(num_classes):
            bank_items = self.memory_bank.get(c, [])

            if len(bank_items) > 0:
                # 提取该类别下所有特征并堆叠
                feats = torch.stack(
                    [item[1] for item in bank_items],
                    dim=0
                )  # (N_c, D)

                # 计算 prototype（均值）
                proto = feats.mean(dim=0)

                # L2 归一化
                prototypes[c] = F.normalize(proto, p=2, dim=0)

            else:
                pass
        return prototypes

    def calculate_rectified_score(self, class_level_sim, real_features, num_classes, alpha=0.6, update_bank=True):
        """
        [Test-Time Prototype Rectification] with Global Memory Bank

        Args:
            update_bank (bool): 是否根据当前输入更新 Bank。
                                ZSL Test 时设为 True (Stream)。
                                GZSL Gamma 搜索循环中设为 False (使用预计算好的 Bank)。
        """
        # 1. 更新全局 Bank (如果需要)
        if update_bank:
            self.update_memory_bank(class_level_sim, real_features)

        # 2. 从全局 Bank 构建原型
        feature_dim = real_features.shape[1]
        prototypes = self.get_prototypes_from_bank(num_classes, feature_dim)

        # 3. 计算与全局原型的相似度 (Rectified Score)
        # prototypes: (C, Dim), real_features: (N, Dim) -> (N, C)
        rectified_sim = torch.mm(real_features, prototypes.t())

        # 4. 加权融合
        final_sim = alpha * class_level_sim + (1 - alpha) * rectified_sim

        return final_sim

    def test(self):
        # 仅用于ZSL测试
        self.dit.eval()
        self.dit.requires_grad_(False)

        unseen_class_features = []

        num_ode_steps = self.args.num_steps
        delta_t = 1.0 / num_ode_steps
        timesteps = torch.linspace(0.0, 1.0 - delta_t, num_ode_steps, device=self.accelerator.device)

        # --- 初始化 Memory Bank ---
        # 假设 unseen_labels 是连续的或者我们映射到了 0 ~ N-1
        num_unseen = len(self.unseen_labels)
        self.init_memory_bank(num_classes=num_unseen, max_p=getattr(self.args, 'max_p', 20))

        with torch.no_grad():
            if self.accelerator.is_main_process:
                print(f"Generating pseudo-features for {self.unseen_num} unseen classes (ZSL)...")

            # 1. Flow Matching 生成 Unseen 类的特征库
            for label_idx in tqdm(self.unseen_labels, disable=not self.accelerator.is_main_process):
                labels_ = torch.ones((1)).to(self.accelerator.device) * label_idx
                text_embed = self.text_embed[labels_.long()].to(self.accelerator.device, dtype=self.weight_dtype)
                text_global = text_embed[:, -1, :].repeat(self.args.num_noise, 1)
                text_local = text_embed[:, :-1, :].repeat(self.args.num_noise, 1, 1)

                x_t = torch.randn((self.args.num_noise, 1, self.args.in_channels),
                                  device=self.accelerator.device,
                                  dtype=self.weight_dtype)

                for t in timesteps:
                    t_batch = t.unsqueeze(0).repeat(self.args.num_noise)
                    v_pred = self.dit(x_t, t_batch, text_global, text_local)
                    x_t = x_t + v_pred * delta_t

                unseen_class_features.append(x_t)

            # (N_unseen * Num_noise, Dim)
            generated_features = torch.cat(unseen_class_features, dim=0).squeeze(1)
            generated_features = F.normalize(generated_features, p=2, dim=-1)

            if self.accelerator.is_main_process:
                print("Classifying test samples (ZSL) with Global Prototype Memory Bank...")

            cnt = 0
            num = 0

            # 2. 遍历测试集进行推理 (Streaming Setting)
            for idx, (features, labels) in tqdm(enumerate(self.test_data_loader),
                                                disable=not self.accelerator.is_main_process):
                features = features.to(self.accelerator.device, dtype=self.weight_dtype)  # (B, Dim)
                labels = labels.to(self.accelerator.device).long()

                # 映射 Label 到 0~Unseen_Num-1
                mapped_labels = [self.unseen_label_mapping[label.item()] for label in labels]
                mapped_labels = torch.Tensor(mapped_labels).to(self.accelerator.device).long()

                features_norm = F.normalize(features, p=2, dim=-1)

                # --- 原始分数计算 (Flow Matching) ---
                sim_matrix_noise = torch.mm(features_norm, generated_features.transpose(0, 1))

                batch_size = features.shape[0]
                sim_matrix_class, _ = sim_matrix_noise.view(batch_size, num_unseen, self.args.num_noise).max(dim=2)

                # update_bank=True: 每来一个 Batch，更新一次 Bank
                final_sim = self.calculate_rectified_score(
                    sim_matrix_class,
                    features_norm,
                    num_classes=num_unseen,
                    alpha=0.2,
                    update_bank=True
                )

                best_matches_class_idx = torch.argmax(final_sim, dim=-1)

                cnt += torch.sum(best_matches_class_idx == mapped_labels)
                num += len(labels)

        zsl_accuracy = float(cnt) / num
        return zsl_accuracy

    def test_gzsl(self):
        """
        Generalized Zero-Shot Learning (GZSL) 测试
        """
        self.dit.eval()
        self.dit.requires_grad_(False)

        end = 0.111
        start = 0.108
        step = (end - start) / (31 - 1)
        calibration_factor_list = [start + i * step for i in range(31)]

        all_class_features = []
        num_ode_steps = self.args.num_steps
        delta_t = 1.0 / num_ode_steps
        timesteps = torch.linspace(0.0, 1.0 - delta_t, num_ode_steps, device=self.accelerator.device)

        # 全局类别列表
        all_labels_to_gen = np.concatenate((self.seen_labels, self.unseen_labels))
        all_labels_to_gen.sort()
        total_num_classes = len(all_labels_to_gen)
        label_to_idx_map = {l: i for i, l in enumerate(all_labels_to_gen)}

        # --- 初始化 Memory Bank ---
        self.init_memory_bank(num_classes=total_num_classes, max_p=getattr(self.args, 'max_p', 20))

        with torch.no_grad():

            if self.accelerator.is_main_process:
                print(f"Generating pseudo-features for GZSL ({total_num_classes} classes)...")

            for label_idx in tqdm(all_labels_to_gen, disable=not self.accelerator.is_main_process):
                labels_ = torch.ones((1)).to(self.accelerator.device) * label_idx
                text_embed = self.text_embed[labels_.long()].to(self.accelerator.device, dtype=self.weight_dtype)
                text_global = text_embed[:, -1, :].repeat(self.args.num_noise, 1)
                text_local = text_embed[:, :-1, :].repeat(self.args.num_noise, 1, 1)

                x_t = torch.randn((self.args.num_noise, 1, self.args.in_channels),
                                  device=self.accelerator.device, dtype=self.weight_dtype)
                for t in timesteps:
                    t_batch = t.unsqueeze(0).repeat(self.args.num_noise)
                    v_pred = self.dit(x_t, t_batch, text_global, text_local)
                    x_t = x_t + v_pred * delta_t
                all_class_features.append(x_t)

            generated_features = torch.cat(all_class_features, dim=0).squeeze(1)
            generated_features = F.normalize(generated_features, p=2, dim=-1)

            # Mask for Seen classes
            is_seen_col_mask = torch.zeros(total_num_classes, device=self.accelerator.device, dtype=torch.bool)
            for i, lbl in enumerate(all_labels_to_gen):
                if lbl in self.seen_labels:
                    is_seen_col_mask[i] = True

            if self.accelerator.is_main_process:
                print(f"Collecting real test features...")

            real_features_list = []
            real_labels_list = []

            for features, labels in self.gzsl_loader:
                features = features.to(self.accelerator.device, dtype=self.weight_dtype)
                labels = labels.to(self.accelerator.device).long()
                real_features_list.append(features)
                real_labels_list.append(labels)

            all_real_features = torch.cat(real_features_list, dim=0)
            all_real_labels = torch.cat(real_labels_list, dim=0)
            all_real_labels_idx = torch.tensor(
                [label_to_idx_map[l.item()] for l in all_real_labels],
                device=self.accelerator.device
            ).long()
            all_real_features = F.normalize(all_real_features, p=2, dim=-1)


            base_sim_matrix_noise = torch.mm(all_real_features, generated_features.transpose(0, 1))
            N_test = all_real_features.shape[0]
            base_sim_matrix_class, _ = base_sim_matrix_noise.view(N_test, total_num_classes, self.args.num_noise).max(
                dim=2)

            if self.accelerator.is_main_process:
                print("Populating Global Memory Bank with test data (Simulating Streaming)...")

            self.update_memory_bank(base_sim_matrix_class, all_real_features)

            # ==========================================
            # 阶段 4: 循环搜索最佳 Gamma
            # ==========================================
            best_acc_seen = 0.0
            best_acc_unseen = 0.0
            best_H = -1.0
            best_gamma = 0.0

            seen_labels_tensor = torch.tensor(self.seen_labels, device=self.accelerator.device)

            for i, gamma in enumerate(calibration_factor_list):
                # 4.1 应用 Calibrated Stacking
                current_flow_score = base_sim_matrix_class.clone()
                current_flow_score[:, is_seen_col_mask] -= gamma

                final_score = self.calculate_rectified_score(
                    current_flow_score,
                    all_real_features,
                    num_classes=total_num_classes,
                    alpha=0.2,
                    update_bank=False
                )

                predicted_idx = torch.argmax(final_score, dim=-1)
                match = (predicted_idx == all_real_labels_idx)

                is_seen_sample = torch.isin(all_real_labels, seen_labels_tensor)
                is_unseen_sample = ~is_seen_sample

                total_seen = is_seen_sample.sum().item()
                correct_seen = (match & is_seen_sample).sum().item()
                acc_seen = correct_seen / total_seen if total_seen > 0 else 0.0

                total_unseen = is_unseen_sample.sum().item()
                correct_unseen = (match & is_unseen_sample).sum().item()
                acc_unseen = correct_unseen / total_unseen if total_unseen > 0 else 0.0

                if (acc_seen + acc_unseen) > 0:
                    H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
                else:
                    H = 0.0

                if H > best_H:
                    best_H = H
                    best_acc_seen = acc_seen
                    best_acc_unseen = acc_unseen
                    best_gamma = gamma
                elif best_H == 0.0 and H == 0.0:
                    if acc_seen > best_acc_seen:
                        best_acc_seen = acc_seen

            if self.accelerator.is_main_process:
                print(f"Best Gamma Found: {best_gamma:.4f} -> H: {best_H:.4f}")

            return best_acc_seen, best_acc_unseen, best_gamma, best_H