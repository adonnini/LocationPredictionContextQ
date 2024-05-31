import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import math

from typing import Dict
from typing import Optional,Final,cast
import pandas as pd
import numpy as np
from easydict import EasyDict as edict
from torch.nn import Dropout


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[: token_embedding.size(0), :])


class TemporalEmbedding(nn.Module):
    def __init__(self, d_input, emb_info="all"):
        super(TemporalEmbedding, self).__init__()

        self.emb_info = emb_info
        # quarter of an hour
        self.minute_size = 4
        hour_size = 24
        weekday = 7

        #AD ADDED
        self.time_embed = nn.Embedding(self.minute_size * hour_size, d_input)

        print("LocationPrediction - embed.py - TemporalEmbedding - emb_info - 0 - ",emb_info)

        if self.emb_info == "all":
            self.minute_embed = nn.Embedding(self.minute_size, d_input)
            self.hour_embed = nn.Embedding(hour_size, d_input)
            self.weekday_embed = nn.Embedding(weekday, d_input)
        elif self.emb_info == "time":
            self.time_embed = nn.Embedding(self.minute_size * hour_size, d_input)
        elif self.emb_info == "weekday":
            self.weekday_embed = nn.Embedding(weekday, d_input)

    def forward(self, time, weekday):

        print("LocationPrediction - embed.py - TemporalEmbedding - forward - time - ",time)
        print("LocationPrediction - embed.py - TemporalEmbedding - forward - weekday - ",weekday)

        if self.emb_info == "all":
            hour = torch.div(time, self.minute_size, rounding_mode="floor")
            minutes = time % 4

            minute_x = self.minute_embed(minutes)
            hour_x = self.hour_embed(hour)
            weekday_x = self.weekday_embed(weekday)

            return hour_x + minute_x + weekday_x
        elif self.emb_info == "time":
            return self.time_embed(time)
        elif self.emb_info == "weekday":
            return self.weekday_embed(weekday)


class POINet(nn.Module):
    def __init__(self, poi_vector_size, out):
        super(POINet, self).__init__()

        self.buffer_num = 11

        # 11 -> poi_vector_size*2 -> 11
        if self.buffer_num == 11:
            self.linear1 = torch.nn.Linear(self.buffer_num, poi_vector_size * 2)
            self.linear2 = torch.nn.Linear(poi_vector_size * 2, self.buffer_num)
            self.dropout2 = nn.Dropout(p=0.1)
            self.norm1 = nn.LayerNorm(self.buffer_num)

            # 11*poi_vector_size -> poi_vector_size
            self.dense = torch.nn.Linear(self.buffer_num * poi_vector_size, poi_vector_size)
            self.dropout_dense = nn.Dropout(p=0.1)

        # poi_vector_size -> poi_vector_size*4 -> poi_vector_size
        self.linear3 = torch.nn.Linear(poi_vector_size, poi_vector_size * 4)
        self.linear4 = torch.nn.Linear(poi_vector_size * 4, poi_vector_size)
        self.dropout3 = nn.Dropout(p=0.1)
        self.dropout4 = nn.Dropout(p=0.1)
        self.norm2 = nn.LayerNorm(poi_vector_size)

        # poi_vector_size -> out
        self.fc = nn.Linear(poi_vector_size, out)

    def forward(self, x):
        # first
        if self.buffer_num == 11:
            x = self.norm1(x + self._ff_block(x))
        # flat
        x = x.view([x.shape[0], x.shape[1], x.shape[2] * x.shape[3]])
        if self.buffer_num == 11:
            x = self.dropout_dense(F.relu(self.dense(x)))
        # second
        x = self.norm2(x + self._dense_block(x))
        return self.fc(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(F.relu(self.linear1(x)))
        return self.dropout2(x)

    def _dense_block(self, x: Tensor) -> Tensor:
        x = self.linear4(self.dropout3(F.relu(self.linear3(x))))
        return self.dropout4(x)


class AllEmbedding(nn.Module):
    def __init__(self, d_input, config, total_loc_num, if_pos_encoder=True, emb_info="all", emb_type="add") -> None:
        super(AllEmbedding, self).__init__()
        # emberdding layers
        self.d_input = d_input
        self.emb_type = emb_type

        # location embedding
        if self.emb_type == "add":
            self.emb_loc = nn.Embedding(total_loc_num, d_input)
        else:
            self.emb_loc = nn.Embedding(total_loc_num, d_input - config.time_emb_size)

        # time is in minutes, possible time for each day is 60 * 24 // 30
        self.if_include_time = config.if_embed_time
        if self.if_include_time:
            if self.emb_type == "add":
                self.temporal_embedding = TemporalEmbedding(d_input, emb_info)
            else:
                self.temporal_embedding = TemporalEmbedding(config.time_emb_size, emb_info)

        # duration is in minutes, possible duration for two days is 60 * 24 * 2// 30
        self.if_include_duration = config.if_embed_duration
        if self.if_include_duration:
            self.emb_duration = nn.Embedding(60 * 24 * 2 // 30, d_input)

        # poi
        self.if_include_poi = config.if_embed_poi
        if self.if_include_poi:
            self.poi_net = POINet(config.poi_original_size, d_input)

        # position encoder for transformer
        self.if_pos_encoder = if_pos_encoder
        self.dropout = nn.Dropout(0.1)

        if self.if_pos_encoder:
            self.pos_encoder = PositionalEncoding(d_input, dropout=0.1)
        else:
            self.dropout = nn.Dropout(0.1)

    # def forward(self, src, tensor_len, tensor_user, tensor_time, tensor_diff, tensor_duration, tensor_weekday) -> Tensor:
    # def forward(self, src, context_dict) -> Tensor:
    # def forward(self, src, context_dict: Dict[str, torch.Tensor]) -> Tensor:
    # def forward(self, src, context_dict: torch.Tensor=torch.Tensor()) -> Tensor:

    def forward(self, src, tensor_len, tensor_user, tensor_time, tensor_diff, tensor_duration, tensor_weekday) -> Tensor:
        emb = self.emb_loc(src)

        # import tensorflow as tf
        # context_dict_tensor = tf.convert_to_tensor(list(context_dict.values()))
        # print("LocationPrediction - embed.py - AllEmbedding.forward - context_dict_tensor - 0 - ",context_dict_tensor)

        # from tensordict import TensorDict
        # data = TensorDict({
        #     "len": context_dict["len"],
        #     "user": context_dict["user"],
        #     "time": context_dict["time"],
        #     "diff": context_dict["diff"],
        #     "duration": context_dict["duration"],
        #     "weekday": context_dict["weekday"],
        # }, batch_size=[context_dict["time"].size(dim=0), context_dict["time"].size(dim=1)])
        # print("LocationPrediction - embed.py - AllEmbedding.forward - data - 0 - ",data)

        # # Creates a dataframe with keys as index and values as cell values.
        # df = pd.DataFrame(context_dict)
        # # Create a new set of index from min and max of the dictionary keys.
        # new_index = np.arange(int(df.index.min()),
        #                       int(df.index.max())).astype(str)
        # # Add the new index to the existing index and fill the nan values with 0, take a transpose of dataframe.
        # new_df = df.reindex(new_index).fillna(0).T.astype(int)
        #
        # context_dict_tensor = df.convert_to_tensor(new_df.values)

        # contextDict = edict(context_dict)

        # print("LocationPrediction - embed.py - AllEmbedding.forward - context_dict - 0 - ",context_dict)
        # print("LocationPrediction - embed.py - AllEmbedding.forward - contextDict - 0 - ",contextDict)

        print("LocationPrediction - embed.py - AllEmbedding.forward - tensor_time - ",tensor_time)
        print("LocationPrediction - embed.py - AllEmbedding.forward - tensor_weekday - ",tensor_weekday)

        input_time = tensor_time
        input_weekday = tensor_weekday
        # input_time = contextDict.time
        # input_weekday = contextDict.weekday
        # input_time = context_dict["time"]
        # input_weekday = context_dict["weekday"]
        # input_duration = context_dict["duration"]
        # input_poi = context_dict["poi"]

        if self.if_include_time:
            if self.emb_type == "add":
                # tmp_temporal_embedding = self.temporal_embedding(input_time, input_weekday)
                # tmp_list = self.temporal_embedding(input_time, input_weekday).tolist()
                # tmp_tensor = torch.tensor(tmp_list)
                # emb = emb + tmp_tensor
                temporal_embedding_time_weekday = self.temporal_embedding(input_time, input_weekday)
                if temporal_embedding_time_weekday is not None:
                    emb = emb + temporal_embedding_time_weekday
                # emb = emb + self.temporal_embedding(input_time, input_weekday)
                # emb = emb + self.temporal_embedding(context_dict["time"], context_dict["weekday"])
            else:
                temporal_embedding_time_weekday = self.temporal_embedding(input_time, input_weekday)
                if temporal_embedding_time_weekday is not None:
                    emb = torch.cat([emb, temporal_embedding_time_weekday], dim=-1)
                # emb = torch.cat([emb, self.temporal_embedding(tensor_time, tensor_weekday)], dim=-1)
                # emb = torch.cat([emb, self.temporal_embedding(context_dict["time"], context_dict["weekday"])], dim=-1)

        if self.if_include_duration:
            emb = emb + self.emb_duration(tensor_duration)
            # emb = emb + self.emb_duration(context_dict["duration"])

        # if self.if_include_poi:
        #     emb = emb + self.poi_net(context_dict["poi"])

        if self.if_pos_encoder:
            return self.pos_encoder(emb * math.sqrt(self.d_input))
        else:
            return self.dropout(emb)
