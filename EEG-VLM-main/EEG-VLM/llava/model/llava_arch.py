# Modified from LLaVA: https://github.com/haotian-liu/LLaVA.git

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np
import os

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

"""
    为大语言模型准备多模态输入
"""
class LlavaMetaForCausalLM(ABC):

    # Log the current training batch
    # 初始化状态控制：用于追踪当前训练或评估的批次索引，以确保特征向量与输入图像按顺序一一对应
    def __init__(self):
        self.current_batch_idx = 0

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    # 通用视觉特征
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    # 专门生理特征
    """
    原始代码：
    def encode_resnet_features(self, resnet_features_path):
        resnet_features_np = np.load(resnet_features_path)
        # This is used during training
        resnet_features_tensor = torch.from_numpy(resnet_features_np).to(self.device,
                                                                        dtype=torch.bfloat16)  # (N, 1, 1024)
        # This is used during evaluation
        # 精度转换
        # resnet_features_tensor = torch.from_numpy(resnet_features_np).to(self.device,
        #                                                                  dtype=torch.half)  # (N, 1, 1024)
        # 对齐与投影
        resnet_features = self.get_model().mm_projector(resnet_features_tensor)  # (N, 1, 1024)->(N, 1, 5120)
        return resnet_features
    """
    def encode_resnet_features(self, resnet_features_path):
        """
        加载 EEG ResNet 高级特征，并将其投影到 LLM 的特征空间。
        """
        # 1. 加载特征
        resnet_features_np = np.load(resnet_features_path)
        
        # 2. 动态获取当前模型的真实计算精度 (自动适配训练的 bf16 或评估的 fp16)
        target_dtype = next(self.get_model().mm_projector.parameters()).dtype
        
        # 3. 转换为 Tensor，并对齐设备与精度
        resnet_features_tensor = torch.from_numpy(resnet_features_np).to(
            device=self.device, 
            dtype=target_dtype
        )  # 形状: (N, 1, 1024)
        
        # 4. 对齐与投影
        resnet_features = self.get_model().mm_projector(resnet_features_tensor)  # 形状: (N, 1, 1024) -> (N, 1, 5120)
        
        return resnet_features

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        # 获取模型中负责处理图像的视觉编码器
        vision_tower = self.get_vision_tower()
        # print("images.shape:", images.shape)  # torch.Size([16, 2, 3, 336, 336])
        per_device_train_batch_size = images.shape[0] #确定批次大小
        # print("input_ids.shape:", input_ids.shape)
        # Under certain conditions (e.g., when processing a vision tower, missing images, or when the input sequence length is 1),
        # the attention_mask and position_ids are adjusted accordingly, and the modified tensors are returned.
        # 模型没有视觉模块（纯文本模型）或当前输入中没有图像数据
        if vision_tower is None or images is None or input_ids.shape[1] == 1:    #input_ids.shape：[batch_size, sequence_length]
            # 处理自回归预测（生成阶段）时的输入对齐，确保在逐个生成单词（Token）时，模型能正确识别当前 Token 的位置，并跳过重复的图像特征提取过程 
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                # 计算目标形状
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                # 动态调整掩码与位置偏移
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                # 计算位置 ID
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # 将输入的原始图像张量转换为大模型可以理解的高维特征向量
        if type(images) is list or images.ndim == 5:
            image_features = torch.stack([self.encode_images(img) for img in images], dim=0).to(
                self.device)  # [16, 2, 576, 5120]
        else:
            image_features = self.encode_images(images).to(self.device)
            # print("final_image_features.shape", image_features.shape)

        # Concatenate by batch to obtain the current batch and the corresponding resnet_features
        # Adjust this according to your per_device_train_batch_size
        # 根据当前的批次大小（Batch Size）来决定加载哪一份预先提取的 EEG 特征文件，并从中切片取出当前 Batch 所需的数据
        #训练模式
        if per_device_train_batch_size == 16:
            # Get the path of the directory where the current file is located
            current_file_path = os.path.abspath(__file__)
            current_dir = os.path.dirname(current_file_path)

            # Build the relative path for the train_high_level_shuffled_features.npy file
            resnet_features_path = os.path.join(current_dir, '..', '..', 'sleep_data',
                                                'train_high_level_shuffled_features.npy')
            resnet_features = self.encode_resnet_features(
                resnet_features_path)  # (N, 1, 1024)->(N, 1, 5120)
        # Perform evaluation
        # 评估时，使用预计算的特征
        elif per_device_train_batch_size == 1:
            # Get the path of the directory where the current file is located
            current_file_path = os.path.abspath(__file__)
            current_dir = os.path.dirname(current_file_path)

            # Build the relative path for the eval_high_level_features.npy file
            resnet_features_path = os.path.join(current_dir, '..', '..', 'sleep_data', 'eval_high_level_features.npy')
            resnet_features = self.encode_resnet_features(resnet_features_path)  # (N, 1, 1024)->(N, 1, 5120)
        else:
            print("The per_device_train_batch_size is incorrect. Current per_device_train_batch_size:",
                  per_device_train_batch_size)
            exit()
        # 从预先加载的巨大特征矩阵中，精准“切出”当前 Batch 所需数
        # print("resnet_features.shape:", resnet_features.shape)  # torch.Size([N, 1, 5120])
        batch_start_idx = self.current_batch_idx * per_device_train_batch_size
        batch_end_idx = batch_start_idx + per_device_train_batch_size
        resnet_features_batch = resnet_features[batch_start_idx:batch_end_idx].to(self.device)

        # Update current_batch_idx
        self.current_batch_idx += 1
        if self.current_batch_idx * per_device_train_batch_size >= resnet_features.shape[0]:
            self.current_batch_idx = 0  # Reset for next epoch if needed

        print("self.current_batch_idx:", self.current_batch_idx)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError


        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        # The main purpose of this code is to handle None values in the input by using default placeholder tensors.
        # This approach simplifies the subsequent code logic and avoids repeatedly checking for the presence of None.
        # 备份原始状态、输入的初始化与缺省值填充
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        # 利用注意力掩码（attention_mask）将输入数据中的填充（Padding）部分去除，只保留有效数据
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # This code processes the input_ids in each batch, checks whether image tokens are present, and handles image features and labels accordingly.
        # Specifically, it replaces image tokens with image features, and adjusts the input embeddings and labels based on the presence of image tokens.
        # 遍历每个样本，将文本中的 <image> 占位符替换为实际的特征向量，并在此过程中实现“通用视觉特征”与“脑电高层特征”的物理融合
        new_input_embeds = []
        new_labels = []
        cur_resnet_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_image_idx = 0
            # Count the number of image tokens (IMAGE_TOKEN_INDEX) in the current input.
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # print("num_images", num_images)

            # 特征维度检查与提取
            if image_features.ndim == 4:
                batch_image_features = image_features[batch_idx]
            else:
                batch_image_features = image_features
            
            # 纯文本处理
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)  #文本编码
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            # Process inputs that contain image tokens.
            # 把文本部分（Text）从图像占位符（Image Token）之间剥离出来，单独转换成向量（Embedding），以便稍后与真正的图像特征进行重组
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]]
            # print("image_token_indices", image_token_indices)   # [-1, 12, 78]
            # Split input_ids and labels.
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            # 切割文本片段
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            # Obtain the cur_input_embeds_no_im and split_sizes.
            split_sizes = [x.shape[0] for x in cur_labels_noim]  # [12, 65]
            # 批量向量化
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim)) 
            # 还原切割
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            # Construct cur_new_input_embeds and cur_new_labels
            # 将切割好的文本片段（Text Embeddings）与两种不同的视觉特征（CLIP 特征和 ResNet 特征）像“三明治”一样交替拼接起来，构造出最终输入给大模型的序列
            for i in range(num_images + 1):
                # 循环结构：交替拼接
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                # Process low-level and high-level visual features separately
                # 偶数位置（i % 2 == 0）：处理第一张图
                if i < num_images:
                    if i % 2 == 0:
                        cur_image_features = batch_image_features[cur_image_idx]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(
                            torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device,
                                       dtype=cur_labels.dtype))
                    # 奇数位置（else）：处理第二张图
                    else:
                        # 取出 ResNet 特征
                        cur_resnet_features = resnet_features_batch[cur_resnet_idx]
                        cur_resnet_idx += 1
                        # 维度扩展 (Repeat)
                        # ResNet 提取的是一个全局向量 [1, 1024]，而 CLIP 提取的是序列特征 [576, 1024]。
                        # 为了能相加，必须把 ResNet 向量复制 576 次，使其形状变为 [576, 1024]
                        cur_resnet_features = cur_resnet_features.repeat(576, 1)
                        # 回溯上一张图的特征
                        # 这里取的是 cur_image_idx - 1，也就是刚刚处理过的 Image1 的 CLIP 特征。
                        cur_image_features = batch_image_features[cur_image_idx - 1]
                        # 特征融合 (Addition)
                        # 将“ResNet 生理特征”直接加到“CLIP 视觉特征”上。
                        cur_resnet_features = cur_resnet_features + cur_image_features
                        # 插入融合后的特征
                        cur_new_input_embeds.append(cur_resnet_features)
                        cur_new_labels.append(
                            torch.full((cur_resnet_features.shape[0],), IGNORE_INDEX, device=cur_labels.device,
                                       dtype=cur_labels.dtype))

            # Concatenate the new embeddings and labels, then append them to new_input_embeds and new_labels.
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        # 序列截断（Sequence Truncation），用于防止处理后的多模态输入序列长度超过模型允许的最大限制
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)  #确定画布尺寸
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype,
                                       device=new_labels[0].device) #标签填充
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device) #掩码填充
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device) #位置编码填充

        # Pad the embedding vectors and labels to a consistent length, and select the padding direction (left or right) based on the configuration settings.
        # 执行物理填充（Padding），将之前处理好的长短不一的样本（Ragged Tensors）真正地排列到一个整齐的矩阵中，以便送入 GPU 进行并行计算
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            # 左填充,推理/生成阶段
            """
                Labels：把真实标签填入最后部分，前面保留初始化时的 IGNORE_INDEX。
                Mask：把最后部分的掩码设为 True（可见），前面保留 False（屏蔽）。
                Position IDs：从 0 到 cur_len 重新生成位置索引，填入最后部分。
            """
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                              device=position_ids.device)
            # 右填充 (else) —— 训练时的默认路径
            else:
                """
                    Labels：真实标签填在前面，后面留白。
                    Mask：前面设为 True，后面填充部分保持 False（这样 Attention 机制就会忽略后面的 0）。
                    Position IDs：在前面填入 0, 1, 2...。
                """
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                             device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # 负责将之前处理好的列表转换成张量（Tensor），并根据最原始的输入状态决定最终要返回哪些数据
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
