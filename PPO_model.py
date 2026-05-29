import copy
import math
import os
import random
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli

from graph.hgnn import GATedge, MLPsim
from mlp import MLPCritic, MLPActor, MLPActorHigh

def stack_tensors(tensors_list, pad_dim=1):
    # Determine the number of dimensions
    num_dims = len(tensors_list[0].shape)

    # Find the maximum shape in each dimension
    max_second_dim = max(tensor.shape[1] for tensor in tensors_list)
    max_third_dim = max(tensor.shape[2] for tensor in tensors_list) if num_dims > 2 else 0
    max_fourth_dim = max(tensor.shape[3] for tensor in tensors_list) if num_dims > 3 else 0

    padded_tensors = None

    if num_dims == 4 and pad_dim == 1:
        padded_tensors = [torch.nn.functional.pad(tensor, (
        0, 0, 0, 0, 0, max_third_dim - tensor.shape[2], 0, max_second_dim - tensor.shape[1])) for tensor in
                          tensors_list]
    elif num_dims == 4 and pad_dim == 2:
        padded_tensors = [torch.nn.functional.pad(tensor, (
        0, max_fourth_dim - tensor.shape[3], 0, max_third_dim - tensor.shape[2], 0, max_second_dim - tensor.shape[1]))
                          for tensor in tensors_list]
    elif num_dims == 3:
        if pad_dim == 1:
            padded_tensors = [torch.nn.functional.pad(tensor, (0, 0, 0, max_second_dim - tensor.shape[1])) for tensor in
                              tensors_list]
        elif pad_dim == 2:
            padded_tensors = [torch.nn.functional.pad(tensor, (
            0, max_third_dim - tensor.shape[2], 0, max_second_dim - tensor.shape[1])) for tensor in tensors_list]

    stacked_tensors = torch.stack(padded_tensors, dim=0)
    return stacked_tensors

class Memory:
    def __init__(self):
        self.job_logprobs = []
        self.action_probs = []  # used for negotiation process to select new action

        self.rewards = []
        self.is_terminals = []

        self.action_indexes = []
        self.job_action_indexes = []

        self.ope_ma_adj_shared = []

        self.ope_pre_adj_shared = []

        self.ope_sub_adj_shared = []

        self.raw_opes_shared = []
        self.raw_mas_shared = []
        self.proc_time_shared = []

        self.jobs_gather = []
        self.jobs_gather_proc = []
        self.eligible = []
        self.nums_opes_shared = []

        self.attention_coeffs = []

    def clear_memory(self):
        del self.job_logprobs[:]
        del self.action_probs[:]

        del self.rewards[:]
        del self.is_terminals[:]

        del self.action_indexes[:]
        del self.job_action_indexes[:]

        del self.ope_ma_adj_shared[:]

        del self.ope_pre_adj_shared[:]

        del self.ope_sub_adj_shared[:]

        del self.raw_opes_shared[:]
        del self.raw_mas_shared[:]
        del self.proc_time_shared[:]

        del self.jobs_gather[:]
        del self.jobs_gather_proc[:]
        del self.eligible[:]
        del self.nums_opes_shared[:]

        del self.attention_coeffs[:]

    def clear_action_probs(self):
        del self.action_probs[:]

    def clear_attention(self):
        del self.attention_coeffs[:]


class MLPs(nn.Module):
    '''
    MLPs in operation node embedding
    '''
    def __init__(self, W_sizes_ope, hidden_size_ope, out_size_ope, num_head, dropout, in_size_ma, out_size_ma):
        '''
        The multi-head and dropout mechanisms are not actually used in the final experiment.
        :param W_sizes_ope: A list of the dimension of input vector for each type,
        including [machine, operation (pre), operation (sub), operation (self)]
        :param hidden_size_ope: hidden dimensions of the MLPs
        :param out_size_ope: dimension of the embedding of operation nodes
        '''
        super(MLPs, self).__init__()
        self.in_sizes_ope = W_sizes_ope
        self.hidden_size_ope = hidden_size_ope
        self.out_size_ope = out_size_ope
        self.num_head = num_head
        self.dropout = dropout
        self.in_size_ma = in_size_ma
        self.out_size_ma = out_size_ma
        self.gnn_layers = nn.ModuleList()

        # A total of five MLPs and MLP_0 (self.project) aggregates information from other MLPs
        self.gnn_layers.append(GATedge((self.in_size_ma, self.in_sizes_ope[0]), self.out_size_ope, self.num_head,
                                         self.dropout, self.dropout, activation=F.elu))

        for i in range(len(self.in_sizes_ope)):
            self.gnn_layers.append(MLPsim(self.in_sizes_ope[i], self.out_size_ope, self.hidden_size_ope, self.num_head,
                                          self.dropout, self.dropout))

        self.project = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.out_size_ope * (len(self.in_sizes_ope)+1), self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.out_size_ope),
        )

    def forward(self, ope_ma_adj_batch, ope_pre_adj_batch, ope_sub_adj_batch, batch_idxes, feats):
        '''
        :param ope_ma_adj_batch: Adjacency matrix of operation and machine nodes
        :param ope_pre_adj_batch: Adjacency matrix of operation and pre-operation nodes
        :param ope_sub_adj_batch: Adjacency matrix of operation and sub-operation nodes
        :param batch_idxes: Uncompleted instances
        :param feats: Contains operation, machine and edge features
        '''

        mas_expanded = feats[0].unsqueeze(-3).expand(-1, feats[1].size(1), -1, -1)
        proc_expanded = feats[1]
        mas_proc = torch.cat((mas_expanded, proc_expanded), dim=-1)
        h = ([feats[0], feats[1], feats[2]], feats[2].unsqueeze(-3), feats[2].unsqueeze(-3), feats[2].unsqueeze(-3))
        # Identity matrix for self-loop of nodes
        self_adj = torch.eye(feats[2].size(-2),
                             dtype=torch.int64).unsqueeze(0).expand_as(ope_pre_adj_batch[batch_idxes])

        # Calculate an return operation embedding
        adj = (ope_ma_adj_batch[batch_idxes].unsqueeze(-1), ope_pre_adj_batch[batch_idxes].unsqueeze(-1),
               ope_sub_adj_batch[batch_idxes].unsqueeze(-1), self_adj.unsqueeze(-1))
        MLP_embeddings = []
        for i in range(len(adj)):
            MLP_embeddings.append(self.gnn_layers[i](h[i], adj[i], batch_idxes))
        MLP_embedding_in = torch.cat(MLP_embeddings, dim=-1)
        mu_ij_prime = self.project(MLP_embedding_in)
        #  for no opes?
        mask = torch.all(MLP_embedding_in == 0, dim=2)[:, :, None].expand_as(mu_ij_prime)
        mu_ij_prime[mask] = 0
        return mu_ij_prime


class HGNNScheduler(nn.Module):
    def __init__(self, model_paras):
        super(HGNNScheduler, self).__init__()
        self.device = model_paras["device"]
        self.in_size_ma = model_paras["in_size_ma"]  # Dimension of the raw feature vectors of machine nodes
        self.out_size_ma = model_paras["out_size_ma"]  # Dimension of the embedding of machine nodes
        self.in_size_ope = model_paras["in_size_ope"]  # Dimension of the raw feature vectors of operation nodes
        self.out_size_ope = model_paras["out_size_ope"]  # Dimension of the embedding of operation nodes
        self.hidden_size_ope = model_paras["hidden_size_ope"]  # Hidden dimensions of the MLPs
        self.actor_dim = model_paras["actor_in_dim"]  # Input dimension of sub_actor
        self.critic_dim = model_paras["critic_in_dim"]  # Input dimension of critic
        self.n_latent_actor = model_paras["n_latent_actor"]  # Hidden dimensions of the sub_actor
        self.n_latent_critic = model_paras["n_latent_critic"]  # Hidden dimensions of the critic
        self.n_hidden_actor = model_paras["n_hidden_actor"]  # Number of layers in sub_actor
        self.n_hidden_critic = model_paras["n_hidden_critic"]  # Number of layers in critic
        self.action_dim = model_paras["action_dim"]  # Output dimension of sub_actor

        # len() means of the number of HGNN iterations
        # and the element means the number of heads of each HGNN (=1 in final experiment)
        self.num_heads = model_paras["num_heads"]
        self.dropout = model_paras["dropout"]

        # Operation node embedding
        self.get_operations = nn.ModuleList()
        self.get_operations.append(MLPs([self.in_size_ope, self.in_size_ope, self.in_size_ope],
                                        self.hidden_size_ope, self.out_size_ope, self.num_heads[0], self.dropout, self.in_size_ma, self.out_size_ope))
        for i in range(len(self.num_heads)-1):
            self.get_operations.append(MLPs([self.out_size_ope, self.out_size_ope, self.out_size_ope],
                                            self.hidden_size_ope, self.out_size_ope, self.num_heads[i], self.dropout, self.in_size_ma, self.out_size_ope))

        self.sub_actor = MLPActor(self.n_hidden_actor, self.out_size_ope+self.in_size_ma+2, self.n_latent_actor, self.action_dim).to(self.device)
        self.critic = MLPCritic(self.n_hidden_critic, self.out_size_ope+self.in_size_ma+2, self.n_latent_critic, 1).to(self.device)

    def forward(self):
        '''
        Replaced by separate act and evaluate functions
        '''
        raise NotImplementedError

    '''
        raw_opes: shape: [len(batch_idxes), max(num_opes), in_size_ope]
        raw_mas: shape: [len(batch_idxes), num_mas, in_size_ma]
        proc_time: shape: [len(batch_idxes), max(num_opes), num_mas]
    '''
    def get_normalized(self, raw_opes, raw_mas, proc_time, min_proc, max_proc, max_flex, min_opes, max_opes, state):
        '''
        :param raw_opes: Raw feature vectors of operation nodes
        :param raw_mas: Raw feature vectors of machines nodes
        :param proc_time: Processing time
        :return: Normalized feats, including operations, machines and edges
        '''

        # ope features normalisation
        norm_opes = torch.zeros(size=(raw_opes.shape[0], raw_opes.shape[1], 4))
        start = (raw_opes[:, :, 5] - raw_opes[:, :, 7]).div(1)
        start = torch.max(torch.zeros_like(start), start)
        available_op = 1.0 - torch.min(torch.ones(start.size()), start)
        completion = (raw_opes[:, :, 4] - raw_opes[:, :, 7]).div(1)
        deadlines = (raw_opes[:, :, 6] - raw_opes[:, :, 7]).div(1)
        neighbouring_mas = raw_opes[:, :, 1]
        nums_ope_batch = state.nums_ope_batch.gather(1, state.opes_appertain_batch)
        remaining_opes = raw_opes[:, :, 3]

        non_opes_mask = raw_opes[:,:,6] == 0.0
        start[non_opes_mask] = torch.tensor(float('nan'))
        available_op[non_opes_mask] = torch.tensor(float('nan'))
        completion[non_opes_mask] = torch.tensor(float('nan'))
        deadlines[non_opes_mask] = torch.tensor(float('nan'))
        remaining_opes[non_opes_mask] = torch.tensor(float('nan'))
        neighbouring_mas[non_opes_mask] = torch.tensor(float('nan'))

        max_completion = torch.max(torch.nan_to_num(completion, float('-inf')), dim=1)[0].unsqueeze(1)

        norm_opes[:, :, 0] = (start).div(max_completion)
        norm_opes[:, :, 1] = (completion).div(max_completion)
        norm_opes[:, :, 2] = (deadlines).div(max_completion)
        temp_4 = neighbouring_mas.div(raw_mas.size(1))
        norm_opes[:, :, 3] = temp_4

        # mas features normalisation
        norm_mas = torch.zeros(size=(raw_mas.shape[0], raw_mas.shape[1], 3))
        avail_curr_time_diff = (raw_mas[:, :, 1] - raw_mas[:, :, 3]) # time to avail  avail_time - current_time
        idle_time= torch.where(avail_curr_time_diff < 0.0, torch.abs(avail_curr_time_diff), torch.tensor(0.0))  # idle time
        time_till_available = torch.where(avail_curr_time_diff > 0.0, avail_curr_time_diff, torch.tensor(0.0))  # time till available
        max_idle = torch.max(torch.nan_to_num(idle_time, float('-inf')), dim=1)[0].unsqueeze(1)
        min_idle = torch.min(torch.nan_to_num(idle_time, float('inf')), dim=1)[0].unsqueeze(1)
        diff_idle = max_idle - min_idle
        den_idle = torch.where(diff_idle==0.0, torch.tensor(1.0), diff_idle.float())
        max_till_avail = torch.max(torch.nan_to_num(time_till_available, float('-inf')), dim=1)[0].unsqueeze(1)
        min_till_avail = torch.min(torch.nan_to_num(time_till_available, float('inf')), dim=1)[0].unsqueeze(1)
        diff_till_avail = max_till_avail - min_till_avail
        den_till_avail = torch.where(diff_till_avail == 0.0, torch.tensor(1.0), diff_till_avail.float())

        norm_mas[:, :, 0] = (idle_time - torch.mean(idle_time, dim=1).unsqueeze(1)).div(den_idle)
        norm_mas[:, :, 1] =(time_till_available - torch.mean(time_till_available, dim=1).unsqueeze(1)).div(den_till_avail)

        neighbouring_opes = raw_mas[:, :, 0]
        max_neighbouring_opes = torch.max(neighbouring_opes, dim=1)[0].unsqueeze(1)
        min_neighbouring_opes = torch.min(neighbouring_opes, dim=1)[0].unsqueeze(1)
        dif_neighbouring_opes = max_neighbouring_opes - min_neighbouring_opes
        den_neighbouring_opes = torch.where(dif_neighbouring_opes==0.0, torch.tensor(1.0), dif_neighbouring_opes.float())
        norm_mas[:, :, 2] = (neighbouring_opes - torch.mean(neighbouring_opes, dim=1).unsqueeze(1)).div(den_neighbouring_opes)

        expanded_mask = non_opes_mask.unsqueeze(2).expand_as(proc_time)
        proc_time[expanded_mask] = torch.tensor(float('nan'))
        proc_time_norm = proc_time.div(max_completion.unsqueeze(1))
        edge_norm = torch.zeros(size=(proc_time_norm.shape[0], proc_time_norm.shape[1], proc_time_norm.shape[2], 2))
        edge_norm[:, :, :, 1] = proc_time_norm

        proc_infs = proc_time.masked_fill(proc_time == 0.0, float('inf'))
        proc_neg_infs = proc_time.masked_fill(proc_time == 0.0, float('-inf'))
        max_proc = torch.max(torch.nan_to_num(proc_neg_infs, float('-inf')), dim=2)[0]
        min_proc = torch.min(torch.nan_to_num(proc_infs, float('inf')), dim=2)[0]
        diff_proc = max_proc - min_proc
        den_proc = torch.where(diff_proc == 0.0, torch.tensor(1.0), diff_proc.float()).unsqueeze(-1)

        proc_mean_norm = (proc_time - raw_opes[:,:,2].unsqueeze(2)).div(den_proc)
        edge_norm[:, :, :, 0] = torch.where(proc_time == 0.0, torch.tensor(0.0), proc_mean_norm)

        # convert any nans to zeros - nans happen when 'max' is zero
        norm_opes = torch.nan_to_num(norm_opes, nan=0.0, neginf=0, posinf=1)
        norm_mas = torch.nan_to_num(norm_mas, nan=0.0, neginf=0, posinf=1)
        edge_norm = torch.nan_to_num(edge_norm, nan=0.0, neginf=0, posinf=1)

        return  norm_mas, edge_norm, norm_opes

    def get_global_features(self, state, memories, flag_train=False, flag_filter_machines=False, flag_num_to_filter=5):
        batch_idxes = state.batch_idxes
        # Raw feats
        shared_raw_opes = state.feat_opes_batch.transpose(1, 2)[batch_idxes]
        shared_raw_mas = state.feat_mas_batch[:, :, :].transpose(1, 2)[batch_idxes]
        shared_proc_time = state.proc_times_batch[batch_idxes, :, :]
        min_proc = state.min_proc
        max_proc = state.max_proc
        max_flex = state.max_flex
        min_opes = state.min_opes
        max_opes = state.max_opes

        # Normalize
        shared_features = self.get_normalized(shared_raw_opes, shared_raw_mas, shared_proc_time, min_proc, max_proc, max_flex, min_opes, max_opes, state)

        shared_norm_mas = (copy.deepcopy(shared_features[0]))
        shared_norm_proc = (copy.deepcopy(shared_features[1]))
        shared_norm_opes = (copy.deepcopy(shared_features[2]))

        features_glob = copy.deepcopy(shared_features)
        # L iterations of the HGNN to get embeddings
        h_opes_glob = None

        modified_ope_ma_adj = torch.zeros_like(state.ope_ma_adj_batch)
        top_k = min(state.proc_times_batch.size(-1), flag_num_to_filter)
        flag_filter_machines = (top_k == flag_num_to_filter)
        mask = torch.zeros_like(state.proc_times_batch)

        if flag_filter_machines:
            masked_processing_times = state.proc_times_batch.masked_fill(state.proc_times_batch == 0, float('inf'))
            max_proc, _ = torch.max(state.proc_times_batch, dim=-1)
            min_proc, _ = torch.min(masked_processing_times, dim=-1)
            same_proc = torch.all(max_proc == min_proc)
            top_indices =  torch.zeros(state.proc_times_batch.size(0), state.proc_times_batch.size(1), top_k, dtype=torch.int64)
            if same_proc:
                for b in batch_idxes:
                    for ope in range(state.proc_times_batch.size(1)):
                        non_zero_indices = state.proc_times_batch[b,ope,:].nonzero()
                        if non_zero_indices.numel() > 1:  # numel() gives the total number of elements in the tensor
                            non_zero_indices = non_zero_indices.squeeze()
                        if non_zero_indices.size(0) > top_k:
                            avail_time = state.feat_mas_batch[b, 1, non_zero_indices]
                            _, top_indices_temp = torch.topk(avail_time, k=top_k, dim=-1, largest=False)
                            selected_temp = non_zero_indices[top_indices_temp]
                            selected_indices = torch.randperm(non_zero_indices.size(0))[
                                               :top_k]  # Random permutation of indices
                            selected = non_zero_indices[selected_indices]
                            top_indices[b, ope, :] = selected_temp
                        else:
                            padding_size = top_k - non_zero_indices.size(0)
                            selected = F.pad(non_zero_indices, (0, padding_size), "constant", 0)
                            top_indices[b, ope, :] = selected
            else:
                _, top_indices = torch.topk(masked_processing_times, k=top_k, dim=-1, largest=False)
            
            mask.scatter_(-1, top_indices, 1)
            modified_ope_ma_adj = mask * state.ope_ma_adj_batch
        else:
            modified_ope_ma_adj = 1.0 * state.ope_ma_adj_batch

        for i in range(len(self.num_heads)):
            # First Stage, operation node embedding
            # shape: [len(batch_idxes), max(num_opes), out_size_ope]
            h_opes_glob = self.get_operations[i](modified_ope_ma_adj, state.ope_pre_adj_batch,
                                                 state.ope_sub_adj_batch, batch_idxes, features_glob)

            # h_mas_glob, attention_coeffs = self.get_machines[i](state.ope_ma_adj_batch, batch_idxes, features_glob)
            features_glob = (features_glob[0], features_glob[1], h_opes_glob)

        h_mas_glob = features_glob[0]
        h_proc_glob = features_glob[1]

        ope_step_batch = state.ope_step_batch
        jobs_gather = ope_step_batch[..., :, None].expand(-1, -1, h_opes_glob.size(-1))[batch_idxes]
        jobs_gather_proc = \
        ope_step_batch[..., :, None, None].expand(-1, -1, h_proc_glob.size(-2), h_proc_glob.size(-1))[batch_idxes]

        if flag_train:
            memories.ope_ma_adj_shared.append(copy.deepcopy(modified_ope_ma_adj[:, :, :]))
            memories.ope_pre_adj_shared.append(copy.deepcopy(state.ope_pre_adj_batch[:, :, :]))
            memories.ope_sub_adj_shared.append(copy.deepcopy(state.ope_sub_adj_batch[:, :, :]))
            memories.nums_opes_shared.append(copy.deepcopy(state.nums_opes_batch))
            memories.raw_opes_shared.append(copy.deepcopy(shared_norm_opes))
            memories.raw_mas_shared.append(copy.deepcopy(shared_norm_mas))
            memories.proc_time_shared.append(copy.deepcopy(shared_norm_proc))
            memories.jobs_gather.append(copy.deepcopy(jobs_gather))
            memories.jobs_gather_proc.append(copy.deepcopy(jobs_gather_proc))
        #memories.attention_coeffs.append(copy.deepcopy(attention_coeffs.squeeze(-1)))

        return h_mas_glob, h_proc_glob, h_opes_glob, modified_ope_ma_adj

    def get_action_prob(self, state, memories, mas, glob_embeddings, ope_ma, flag_train=False):
        '''
        Get the probability of selecting each action in decision-making
        '''
        # instances
        batch_idxes = state.batch_idxes
        ope_step_batch = state.ope_step_batch

        h_mas_glob = glob_embeddings[0]
        h_proc_glob = glob_embeddings[1]
        h_opes_glob = glob_embeddings[2]

        jobs_gather = ope_step_batch[..., :, None].expand(-1, -1, h_opes_glob.size(-1))[batch_idxes]
        jobs_gather_proc = ope_step_batch[..., :, None, None].expand(-1, -1, h_proc_glob.size(-2), h_proc_glob.size(-1))[batch_idxes]
        h_jobs = h_opes_glob.gather(1, jobs_gather)
        h_proc_job = h_proc_glob.gather(1, jobs_gather_proc)
        h_mas = h_mas_glob[:, mas, :]
        h_proc = h_proc_job[:, :, mas]
        h_mas_padding = h_mas.unsqueeze(1).expand(-1, h_jobs.size(1), -1)  # Shape: (batch_size, num_jobs, 2)

        # Matrix indicating whether processing is possible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        eligible_proc = ope_ma[batch_idxes, :, mas].unsqueeze(-1).gather(1, ope_step_batch[..., :, None])
        # Matrix indicating whether machine is eligible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        ma_eligible = ~state.mask_ma_procing_batch[batch_idxes, mas].unsqueeze(-1).unsqueeze(-1).expand_as(eligible_proc)
        # Matrix indicating whether job is eligible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        job_feasible = ~(state.mask_job_procing_batch[batch_idxes] +
                        state.mask_job_finish_batch[batch_idxes]).unsqueeze(-1)
        # job_feasible = ~(state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(h_jobs_padding[..., 0])
        # shape: [len(batch_idxes), num_jobs, num_mas]
        feasible = job_feasible & ma_eligible & (eligible_proc == 1)

        # Input of sub_actor MLP
        # shape: [len(batch_idxes), num_mas, num_jobs, out_size_ma+out_size_ope]
        h_actions = torch.cat((h_mas_padding, h_proc, h_jobs), dim=-1)
        # h_actions_pooled = torch.cat((h_mas_pooled, h_jobs_pooled), dim=-1)

        mask = feasible.flatten(1)
        scores = self.sub_actor(h_actions).flatten(1)
        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)

        # Store data in memory during training
        if flag_train:
            memories.eligible.append(copy.deepcopy(feasible))

        return action_probs, ope_step_batch, feasible

    def act(self, state, memories, mas, shared_feats, ope_ma, flag_train=False, flag_greedy=False):
        eps = torch.finfo(torch.float64).eps
        hi = torch.ones(state.batch_idxes.size())
        job_probs, ope_step_batch, feasible = self.get_action_prob(state, memories, mas, shared_feats, ope_ma, flag_train=flag_train)
        mask = torch.any(feasible, dim=1).squeeze(1)
        batch_idxes = state.batch_idxes[mask]
        hi[batch_idxes] = 0.0

        # pick job index
        job_probs = torch.nan_to_num(job_probs, nan=0.0)
        haiya = job_probs + hi.unsqueeze(-1)
        haiya = haiya / haiya.sum(dim=1, keepdim=True)
        job_dist = Categorical(haiya)
        job_indexes = job_dist.sample()
        # print(f'argmax {(job_probs + eps).argmax(dim=1)}')
        if flag_greedy:
            job_indexes = (job_probs + eps).argmax(dim=1)

        action_indexes = job_indexes

        # Calculate the machine, job and operation index based on the action index
        mas = torch.ones(state.batch_idxes.size()).long() * mas
        jobs = torch.ones(state.batch_idxes.size()).long() * -2
        opes = torch.ones(state.batch_idxes.size()).long() * -2

        jobs[batch_idxes] = action_indexes[batch_idxes].long()

        sub_batch_idxes = torch.nonzero(jobs > -1).squeeze(-1)

        opes[sub_batch_idxes] = ope_step_batch[sub_batch_idxes, jobs[sub_batch_idxes]]

        # Store data in memory during training
        if flag_train:
            memories.job_logprobs.append(job_dist.log_prob(job_indexes))
            memories.action_indexes.append(jobs)
            memories.job_action_indexes.append(job_indexes)
        memories.action_probs.append(job_probs)  # always save so can extract for use in single act

        return torch.stack((opes, mas, jobs), dim=1).t()

    def single_act(self, state, memories, mas, batch_idx, allocated_jobs, flag_greedy=False):
        eps = torch.finfo(torch.float64).eps
        job_probs = memories.action_probs[mas][batch_idx]

        # mask out job assigned during negotiation
        job_eligible = ~(allocated_jobs[batch_idx] + state.mask_job_finish_batch[batch_idx])
        job_probs = torch.where(job_eligible, job_probs, torch.tensor(0.0, dtype=torch.float))

        job_probs = torch.nan_to_num(job_probs, nan=0.0)
        job_probs = job_probs / job_probs.sum(dim=0, keepdim=True)

        try:
            dist = Categorical(job_probs)
            job_indexes = dist.sample()
            if flag_greedy:
                job_indexes = (job_probs + eps).argmax()
        except Exception:
            job_indexes = torch.tensor(-2)

        # Calculate the machine, job and operation index based on the action index
        mas = torch.ones(1).long() * mas
        jobs = torch.ones(1).long() * -2
        opes = torch.ones(1).long() * -2

        jobs[0] = job_indexes.long()

        ope_step_batch = state.ope_step_batch
        if jobs[0] > -1:
            opes[0] = ope_step_batch[batch_idx, jobs[0]]

        # Store data in memory during training
        # if flag_train:
        #     idx = len(memories.job_logprobs) - num_mas + mas
        #     memories.job_action_indexes[idx][batch_idx] = job_indexes.long()  # only selected action changes
        #     memories.action_indexes[idx][batch_idx] = job_indexes.long()

        return torch.stack((opes, mas, jobs), dim=1).t()

    def get_actions(self, num_mas, state, memories, shared_feats, ope_ma, flag_train=False, flag_greedy=False,
                    negotiate_rule=None):
        group_actions = []
        allocated_jobs = copy.deepcopy(state.mask_job_procing_batch)
        for ma in range(num_mas):
            actions = self.act(state, memories, ma, shared_feats, ope_ma, flag_train, flag_greedy)
            group_actions.append(actions)
            jobs = actions[2, :]
            sub_batch_idxes = torch.nonzero(jobs > -1).squeeze(-1)
            allocated_jobs[sub_batch_idxes, jobs[sub_batch_idxes]] = True
        duplicates = self.find_duplicate_jobs(group_actions)
        dup = bool(duplicates)
        while bool(duplicates):
            # todo: set negotiation strategy here
            unassigned_machines = self.negotiate(duplicates, state, rule=negotiate_rule)
            for batch_idx, machines in unassigned_machines.items():
                for ma in machines:
                    actions = self.single_act(state, memories, ma, batch_idx, allocated_jobs, flag_greedy)
                    group_actions[ma][0, :][batch_idx] = actions[0, :]
                    group_actions[ma][1, :][batch_idx] = actions[1, :]
                    group_actions[ma][2, :][batch_idx] = actions[2, :]
            for ma in range(num_mas):
                actions = group_actions[ma]
                jobs = actions[2, :]
                sub_batch_idxes = torch.nonzero(jobs > -1).squeeze(-1)
                allocated_jobs[sub_batch_idxes, jobs[sub_batch_idxes]] = True
            duplicates = self.find_duplicate_jobs(group_actions)
        return group_actions, dup

    def negotiate(self, duplicates, state, rule="SPT"):
        unassigned_dict = {batch_idx: [] for batch_idx in duplicates}
        for batch_idx, batch_jobs in duplicates.items():
            unassigned_machines = []
            for job_idx, machines in batch_jobs.items():
                # print(f'negotiate batch {batch_idx} job {job_idx} between machines {machines}')
                ope_step_batch = state.ope_step_batch
                ope = ope_step_batch[batch_idx, job_idx]
                sorted_indices = torch.arange(state.feat_mas_batch.size(2)).unsqueeze(0)

                assigned_ma = None

                if rule == "SPT":
                    # sort machine by SPT
                    ope_proc_time = state.proc_times_batch[batch_idx, ope, :]
                    sorted_indices = torch.argsort(ope_proc_time, stable=False)
                elif rule == "LPT":
                    # sort machine by LPT
                    ope_proc_time = state.proc_times_batch[batch_idx, ope, :]
                    sorted_indices = torch.argsort(ope_proc_time, stable=False, descending=True)
                elif rule == "EET":
                    avail_time = state.feat_mas_batch[batch_idx, 1, :]
                    sorted_indices = torch.argsort(avail_time, stable=True)
                elif rule == "LLM":
                    util = state.feat_mas_batch[batch_idx, 2, :]
                    sorted_indices = torch.argsort(util, stable=True)
                elif rule == "Mixed":
                    # Sort machines by SPT
                    ope_proc_time = state.proc_times_batch[batch_idx, ope, :]
                    ope_proc_time = torch.where(ope_proc_time == 0.0, float('inf'), ope_proc_time)
                    sorted_indices_spt = torch.argsort(ope_proc_time, stable=True)
                    util = state.feat_mas_batch[batch_idx, 2, :]
                    sorted_indices_llm = torch.argsort(util, stable=True)
                    # Count the number of non-zero processing times (machines that can perform the job)
                    num_available_machines = torch.sum(ope_proc_time != float('inf')).item()
                    sorted_indices = None

                    if num_available_machines <= 2:
                        # If 2 or fewer machines can perform the job, use LLM to select
                        tied_machines = []
                        for j, ma in enumerate(sorted_indices_llm[:-1]):
                            if ma in machines:
                                for k, next_ma in enumerate(sorted_indices_llm[j+1:-1]):
                                    if util[ma] == util[next_ma]:
                                        if ma not in tied_machines:
                                            tied_machines.append(ma)
                                        if next_ma not in tied_machines:
                                            tied_machines.append(next_ma)
                                break

                        if len(tied_machines) > 0:
                            tied_machines_tensor = torch.tensor(tied_machines)
                            ope_proc_time_2 = state.proc_times_batch[batch_idx, ope, tied_machines_tensor]
                            sorted_indices_spt_2 = tied_machines_tensor[torch.argsort(ope_proc_time_2, stable=True)]
                            sorted_indices = sorted_indices_spt_2
                        else:
                            sorted_indices = sorted_indices_llm
                    else:
                        # Check for ties in SPT and resolve using LLM
                        tied_machines = []
                        for j, ma in enumerate(sorted_indices_spt[:-1]):
                            if ma in machines:
                                for k, next_ma in enumerate(sorted_indices_spt[j+1:-1]):
                                    if ope_proc_time[ma] == ope_proc_time[next_ma]:
                                        if ma not in tied_machines:
                                            tied_machines.append(ma)
                                        if next_ma not in tied_machines:
                                            tied_machines.append(next_ma)
                                break

                        if len(tied_machines) > 0:
                            tied_machines_tensor = torch.tensor(tied_machines)
                            util_2 = state.feat_mas_batch[batch_idx, 2, tied_machines_tensor]
                            sorted_indices_llm_2 = tied_machines_tensor[torch.argsort(util_2, stable=True)]
                            sorted_indices = sorted_indices_llm_2
                        else:
                            sorted_indices = sorted_indices_spt

                if assigned_ma is None:
                    for ma in sorted_indices:
                        if ma in machines:
                            assigned_ma = ma
                            break

                unassigned = [m for m in machines if m != assigned_ma]
                unassigned_machines.extend(unassigned)
            unassigned_dict[batch_idx] = unassigned_machines

        return unassigned_dict

    def negotiatev5(self, duplicates, state, memories=None, flag_greedy=False):
        unassigned_dict = {batch_idx: [] for batch_idx in duplicates}
        for batch_idx, batch_jobs in duplicates.items():
            unassigned_machines = []
            for job_idx, machines in batch_jobs.items():
                # print(f'negotiate batch {batch_idx} job {job_idx} between machines {machines}')
                ope_step_batch = state.ope_step_batch
                ope = ope_step_batch[batch_idx, job_idx]
                ope_proc_time = state.proc_times_batch[batch_idx, ope, :]
                sorted_indices = torch.argsort(ope_proc_time)
                assigned_ma = None
                for ma in sorted_indices:
                    if ma in machines:
                        assigned_ma = ma
                        break
                print(f'assigned ma {assigned_ma}')
                unassigned = [m for m in machines if m != assigned_ma]
                unassigned_machines.extend(unassigned)
            unassigned_dict[batch_idx] = unassigned_machines
        return unassigned_dict

    def negotiatev2(self, duplicates, state, memories=None, flag_greedy=False):
        unassigned_dict = {batch_idx: [] for batch_idx in duplicates}  # dictionary of unassigned machines in each batch
        for batch_idx, batch_jobs in duplicates.items():  # iterate through each batch
            unassigned_machines = []
            for job_idx, machines in batch_jobs.items():  # iterate through each job in batch
                # print(f'negotiate batch {batch_idx} job {job_idx} between machines {machines}')
                ope_step_batch = state.ope_step_batch
                ope = ope_step_batch[batch_idx, job_idx]  # get current operation index of job

                ope_proc_time = state.proc_times_batch[batch_idx, ope, :]  # get op processing times
                end_ope = state.end_ope_biases_batch[batch_idx, job_idx]
                # get sum of mean proc time
                remaining_ope_proc_time = torch.sum(torch.sum(state.proc_times_batch[batch_idx, ope+1:end_ope+1, :], dim=1).div(torch.count_nonzero(state.proc_times_batch[batch_idx, ope+1:end_ope+1, :], dim=1)))
                deadline_met = []
                for ma in machines:  # for conflicting machines, get est. completion time and deadline
                    est_completion_time = remaining_ope_proc_time + ope_proc_time[ma]
                    if est_completion_time < state.deadlines_batch[batch_idx, job_idx]:
                        deadline_met.append(ma)
                if len(deadline_met) == 0:  # round 1: Min Completion Time
                    sorted_indices = torch.argsort(ope_proc_time)  # idx of processing time sorted by processing time in ascending order
                    assigned_ma = None
                    for ma in sorted_indices:  # iterate through each idx
                        if ma in machines:  # if the machine is in the list of machines that are in conflict then assign it
                            assigned_ma = ma
                            break
                    unassigned = [m for m in machines if m != assigned_ma]  # remaining machines in conflict list are marked as unassigned
                else:  # round 2: Min Machine Util
                    ma_utils = state.feat_mas_batch[batch_idx, 2, :]
                    sorted_indices = torch.argsort(ma_utils)
                    assigned_ma = None
                    for ma in sorted_indices:  # iterate through each idx
                        if ma in deadline_met:  # if the machine is in the list of machines that are in conflict then assign it
                            assigned_ma = ma
                            break
                    unassigned = [m for m in machines if m != assigned_ma]  # remaining machines in conflict list are marked as unassigned
                unassigned_machines.extend(unassigned)
            unassigned_dict[batch_idx] = unassigned_machines
        return unassigned_dict

    def negotiatev3(self, duplicates, state, memories, flag_greedy=False):
        eps = torch.finfo(torch.float64).eps
        unassigned_dict = {batch_idx: [] for batch_idx in duplicates}
        for batch_idx, batch_jobs in duplicates.items():
            unassigned_machines = []
            for job_idx, machines in batch_jobs.items():
                # print(f'negotiate batch {batch_idx} job {job_idx} between machines {machines}')
                ope_step_batch = state.ope_step_batch
                ope = ope_step_batch[batch_idx, job_idx]

                ope_proc_time = state.proc_times_batch[batch_idx, ope, :]
                sorted_indices = torch.argsort(ope_proc_time)
                # print(f'processing sort: {sorted_indices}')

                attentions2 = memories.attention_coeffs[0][batch_idx, ope, machines]
                attentions_probs = F.softmax(attentions2, dim=0)
                attentions_dist = Categorical(attentions_probs)
                sample_ma = attentions_dist.sample()
                if flag_greedy:
                    sample_ma = (attentions_probs + eps).argmax()
                assigned_ma = machines[sample_ma]
                # print(f'assigned ma {assigned_ma}')
                unassigned = [m for m in machines if m != assigned_ma]
                unassigned_machines.extend(unassigned)
            unassigned_dict[batch_idx] = unassigned_machines
        return unassigned_dict

    def negotiatev4(self, duplicates, state, memories=None, flag_greedy=False):
        unassigned_dict = {batch_idx: [] for batch_idx in duplicates}  # dictionary of unassigned machines in each batch
        for batch_idx, batch_jobs in duplicates.items():  # iterate through each batch
            unassigned_machines = []
            for job_idx, machines in batch_jobs.items():  # iterate through each job in batch
                # print(f'negotiate batch {batch_idx} job {job_idx} between machines {machines}')
                ope_step_batch = state.ope_step_batch
                ope = ope_step_batch[batch_idx, job_idx]  # get current operation index of job

                ope_proc_time = state.proc_times_batch[batch_idx, ope, :]  # get op processing times
                single_avail = []
                for ma in machines:  # for conflicting machines, get machines that only have one job option
                    job_probs = memories.action_probs[ma][batch_idx]
                    no_jobs_avail = torch.count_nonzero(job_probs)
                    if no_jobs_avail <= 1.0:
                        single_avail.append(ma)
                if len(single_avail) == 0:  # round 1: all jobs have another option
                    sorted_indices = torch.argsort(ope_proc_time)  # idx of processing time sorted by processing time in ascending order
                    assigned_ma = None
                    for ma in sorted_indices:  # iterate through each idx
                        if ma in machines:  # if the machine is in the list of machines that are in conflict then assign it
                            assigned_ma = ma
                            break
                    unassigned = [m for m in machines if m != assigned_ma]  # remaining machines in conflict list are marked as unassigned
                else:  # round 2: machines that only have one job available
                    sorted_indices = torch.argsort(ope_proc_time)
                    assigned_ma = None
                    for ma in sorted_indices:  # iterate through each idx
                        if ma in single_avail:  # if the machine is in the list of machines that are in conflict then assign it
                            assigned_ma = ma
                            break
                    unassigned = [m for m in machines if m != assigned_ma]  # remaining machines in conflict list are marked as unassigned
                unassigned_machines.extend(unassigned)
            unassigned_dict[batch_idx] = unassigned_machines
        return unassigned_dict

    def find_duplicate_jobs(self, group_actions):
        # Initialize a dictionary to keep track of the jobs selected by each machine in each batch
        job_dict = {}

        # Loop over each machine
        for machine_idx, machine_data in enumerate(group_actions):
            # Loop over each batch
            for batch_idx in range(machine_data.shape[1]):
                # Extract the jobs selected by this machine in this batch
                job = machine_data[2, batch_idx]
                job = job.item()
                if job < 0:
                    continue
                # Check if any of these jobs have already been selected by another machine in this batch
                if batch_idx in job_dict and job in job_dict[batch_idx]:
                    # If so, add this machine to the list of machines that have selected this job
                    job_dict[batch_idx][job].append(machine_idx)
                else:
                    # If not, create a new list of machines that have selected this job
                    if batch_idx in job_dict:
                        job_dict[batch_idx][job] = [machine_idx]
                    else:
                        job_dict[batch_idx] = {job: [machine_idx]}

        # Extract the batches and jobs where multiple machines have selected the same job
        duplicate_jobs = {}
        for batch_idx, batch_jobs in job_dict.items():
            for job, machines in batch_jobs.items():
                if len(machines) > 1:
                    duplicate_jobs.setdefault(batch_idx, {})[job] = machines

        # Return the result
        return duplicate_jobs

    def evaluate_embedding(self, shared_ope_ma_adj, ope_pre_adj, ope_sub_adj, shared_raw_opes, shared_raw_mas, shared_proc_time):
        batch_idxes = torch.arange(0, shared_ope_ma_adj.size(-3)).long()
        shared_features = (shared_raw_mas, shared_proc_time, shared_raw_opes)

        shared_h_mas = None
        shared_h_opes = None
        for i in range(len(self.num_heads)):
            shared_h_opes = self.get_operations[i](shared_ope_ma_adj, ope_pre_adj, ope_sub_adj, batch_idxes, shared_features)
            shared_features = (shared_features[0], shared_features[1], shared_h_opes)
            if torch.any(torch.isnan(shared_h_opes)):
                print(f'NaNs present')

        shared_h_mas = shared_features[0]
        shared_h_proc = shared_features[1]

        return shared_h_mas, shared_h_proc, shared_h_opes

    def evaluate_state(self, shared_h_mas, shared_h_proc, shared_h_opes, jobs_gather, jobs_gather_proc):

        shared_h_mas_pooled = shared_h_mas.mean(dim=-2)
        h_jobs = shared_h_opes.gather(1, jobs_gather)
        h_proc = shared_h_proc.gather(1, jobs_gather_proc)
        h_jobs_pooled = h_jobs.mean(dim=1)
        h_proc_pooled = h_proc.mean(dim=1)
        h_proc_pooled = h_proc_pooled.mean(dim=1)

        shared_h_pooled = torch.cat((shared_h_mas_pooled, h_proc_pooled, h_jobs_pooled), dim=-1)

        state_values = self.critic(shared_h_pooled)

        state_values = state_values.squeeze().double()
        if state_values.dim() == 0:
            state_values = state_values.unsqueeze(0)

        return state_values

    def evaluate_job(self, shared_h_mas, shared_h_proc, shared_h_opes, jobs_gather, jobs_gather_proc, eligible, job_action_envs, mas):

        h_jobs = shared_h_opes.gather(1, jobs_gather)
        h_proc = shared_h_proc.gather(1, jobs_gather_proc)
        h_jobs_padding = h_jobs
        h_mas = shared_h_mas[:, mas, :]
        h_proc = h_proc[:, :, mas]
        h_mas_padding = h_mas.unsqueeze(1).expand(-1, h_jobs.size(1), -1)
        h_proc_padding = h_proc

        h_actions = torch.cat((h_mas_padding, h_proc_padding, h_jobs_padding), dim=-1)

        scores = self.sub_actor(h_actions).flatten(1)
        mask = eligible.flatten(1)
        scores[~mask] = float('-inf')

        action_probs = F.softmax(scores, dim=1)

        action_probs = torch.nan_to_num(action_probs, nan=0.0)
        action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)

        dist = Categorical(action_probs.squeeze())
        action_logprobs = dist.log_prob(job_action_envs)
        dist_entropys = dist.entropy()
        return action_logprobs, dist_entropys

class PPO:
    def __init__(self, model_paras, train_paras, num_envs=None):
        self.lr_central = train_paras["lr_central"]  # learning rate
        self.lr_job = train_paras["lr_job"]
        self.betas = train_paras["betas"]  # default value for Adam
        self.gamma = train_paras["gamma"]  # discount factor
        self.eps_clip_job = train_paras["eps_clip_job"]  # clip ratio for PPO
        self.K_epochs = train_paras["K_epochs"]  # Update policy for K epochs
        self.A_coeff = train_paras["A_coeff"]  # coefficient for policy loss
        self.vf_coeff = train_paras["vf_coeff"]  # coefficient for value loss
        self.entropy_coeff_job = train_paras["entropy_coeff_job"]  # coefficient for entropy term
        self.num_envs = num_envs  # Number of parallel instances
        self.device = model_paras["device"]  # PyTorch device

        self.policy = HGNNScheduler(model_paras).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        # self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        params_central = list(self.policy.critic.parameters())
        params_job_actor = list(self.policy.get_operations.parameters()) + list(self.policy.sub_actor.parameters())

        self.optimizer_central = torch.optim.Adam(params_central, lr=self.lr_central, betas=self.betas)
        self.optimizer_job_actor = torch.optim.Adam(params_job_actor, lr=self.lr_job, betas=self.betas)

        self.MseLoss = nn.MSELoss()
        current_lr = self.optimizer_central.param_groups[0]['lr']
        print('Learning rate:', current_lr)

    def gae_advantages(self, rewards, values, dones, lamda=0.95):
        T = len(rewards)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        advantages = [0] * T
        returns = [0] * T
        advantage = 0
        for t in reversed(range(T)):
            if dones[t] or (t == T-1):
                delta_t = rewards[t] - values[t]
                advantage = 0
            else:
                delta_t = rewards[t] + values[t + 1] - values[t]
            advantage = delta_t + lamda * advantage
            advantages[t] = advantage
            returns[t] = advantage + values[t]

        advantages = torch.tensor(advantages)
        returns = torch.tensor(returns)
        return advantages, returns

    def discounted_gae_advantages(self, rewards, values, dones, lamda=0.95, gamma=0.95):
        T = len(rewards)
        advantages = [0] * T
        returns = [0] * T
        advantage = 0
        for t in reversed(range(T)):
            if dones[t] or (t == T-1):
                delta_t = rewards[t] - values[t]
                advantage = 0
            else:
                delta_t = rewards[t] + (gamma * values[t + 1]) - values[t]
            advantage = delta_t + (gamma * lamda * advantage)
            advantages[t] = advantage
            returns[t] = advantage + values[t]

        advantages = torch.tensor(advantages)
        returns = torch.tensor(returns)
        return advantages, returns

    def update(self, memory, env_paras, train_paras):
        device = env_paras["device"]
        num_mas = env_paras["num_mas"]
        minibatch_size = train_paras["minibatch_size"]  # batch size for updating
        discount_factor = train_paras["discount_factor"]
        discounting = train_paras["discounting"]

        # Flatten the data in memory (in the dim of parallel instances and decision points)
        old_ope_ma_adj_shared = stack_tensors(memory.ope_ma_adj_shared).transpose(0, 1).flatten(0, 1)
        old_ope_pre_adj_shared = stack_tensors(memory.ope_pre_adj_shared, pad_dim=2).transpose(0, 1).flatten(0, 1)
        old_ope_sub_adj_shared = stack_tensors(memory.ope_sub_adj_shared, pad_dim=2).transpose(0, 1).flatten(0, 1)
        old_raw_opes_shared = stack_tensors(memory.raw_opes_shared).transpose(0, 1).flatten(0, 1)
        old_raw_mas_shared = torch.stack(memory.raw_mas_shared, dim=0).transpose(0, 1).flatten(0, 1)
        old_proc_time_shared = stack_tensors(memory.proc_time_shared, pad_dim=2).transpose(0, 1).flatten(0, 1)
        old_jobs_gather = stack_tensors(memory.jobs_gather).transpose(0, 1).flatten(0, 1)
        old_jobs_gather_proc = stack_tensors(memory.jobs_gather_proc, pad_dim=2).transpose(0, 1).flatten(0, 1)
        old_eligible = stack_tensors(memory.eligible).transpose(0, 1).flatten(0, 1)
        memory_rewards = torch.stack(memory.rewards, dim=0).transpose(0, 1)
        memory_is_terminals = torch.stack(memory.is_terminals, dim=0).transpose(0, 1)
        memory_is_terminal_flatten = memory_is_terminals.flatten()
        old_job_logprobs = torch.stack(memory.job_logprobs, dim=0).transpose(0, 1).flatten(0, 1)
        old_job_action_envs = torch.stack(memory.job_action_indexes, dim=0).transpose(0, 1).flatten(0, 1)
        old_action_envs = torch.stack(memory.action_indexes, dim=0).transpose(0, 1).flatten(0, 1)
        old_nums_opes_shared = torch.stack(memory.nums_opes_shared, dim=0).transpose(0, 1).flatten(0, 1)

        diffs_envs = []
        avg_rewards = 0
        for i in range(self.num_envs):
            norm_rewards = memory_rewards[i]
            norm_rewards = (norm_rewards-norm_rewards.min()) / (norm_rewards.max() - norm_rewards.min())
            norm_rewards = torch.nan_to_num(norm_rewards)
            avg_reward_pi = norm_rewards.mean()
            if not discounting:
                diffs = norm_rewards - avg_reward_pi
            else:
                diffs = norm_rewards
            plotting_avg = memory_rewards[i].sum()
            avg_rewards += plotting_avg
            diffs_envs.append(diffs)
        diffs_envs = torch.cat(diffs_envs)

        critic_loss_epochs = 0
        job_entropy_loss_epochs = 0
        job_policy_loss_epochs = 0

        non_terminal = ~memory_is_terminal_flatten
        non_zero_opes_indices = torch.nonzero(old_nums_opes_shared)
        full_batch_size = non_zero_opes_indices.size(0)
        num_complete_minibatches = math.floor(full_batch_size / minibatch_size)
        r = 0
        if (full_batch_size % minibatch_size) > 0:
            r = 1
        print(f'num_minibatches {num_complete_minibatches+r}')
        chunks = non_zero_opes_indices.chunk(num_complete_minibatches+r)
        # Optimize policy for K epochs:
        for k in range(self.K_epochs):
            for i in range(num_complete_minibatches+r):
                job_actor_cat_loss = None
                job_entropy_cat_loss = None
                job_policy_cat_loss = None

                indxes = chunks[i].squeeze()
                shared_h_mas, shared_h_proc, shared_h_opes = self.policy.evaluate_embedding(old_ope_ma_adj_shared[indxes, :, :],
                                                          old_ope_pre_adj_shared[indxes, :, :],
                                                          old_ope_sub_adj_shared[indxes, :, :],
                                                          old_raw_opes_shared[indxes, :, :],
                                                          old_raw_mas_shared[indxes, :, :],
                                                          old_proc_time_shared[indxes, :, :])

                state_values = self.policy.evaluate_state(shared_h_mas.detach(), shared_h_proc.detach(), shared_h_opes.detach(), old_jobs_gather[indxes, :, :], old_jobs_gather_proc[indxes, :, :])
                if not discounting:
                    advantages, returnss = self.gae_advantages(diffs_envs[indxes], state_values.detach(), memory_is_terminal_flatten[indxes])
                else:
                    advantages, returnss = self.discounted_gae_advantages(diffs_envs[indxes], state_values.detach(), memory_is_terminal_flatten[indxes], gamma=discount_factor)
                value_loss = self.vf_coeff * self.MseLoss(state_values, returnss)

                for j in range(num_mas):

                    indexes_indexes = indxes * num_mas + j
                    ma_old_action_envs = old_action_envs[indexes_indexes]
                    mask_job = ma_old_action_envs > -1  # mask elements that are continuous

                    ma_old_jobs_gather = old_jobs_gather[indxes, :, :][mask_job]
                    ma_old_jobs_gather_proc = old_jobs_gather_proc[indxes, :, :][mask_job]
                    ma_old_eligible = old_eligible[indexes_indexes, :, :][mask_job]
                    ma_old_job_action_envs = old_job_action_envs[indexes_indexes][mask_job]

                    if ma_old_job_action_envs.numel() <= 0:  # if all the actions are contnue
                        continue
                    jobs_logprobs, jobs_dist_entropy = \
                        self.policy.evaluate_job(shared_h_mas[mask_job],
                                                 shared_h_proc[mask_job],
                                                 shared_h_opes[mask_job],
                                                 ma_old_jobs_gather,
                                                 ma_old_jobs_gather_proc,
                                                 ma_old_eligible,
                                                 ma_old_job_action_envs, j)
                    ratios = torch.exp(
                        jobs_logprobs - old_job_logprobs[indexes_indexes][mask_job])

                    surr1 = ratios * advantages[mask_job]
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip_job, 1 + self.eps_clip_job) * advantages[mask_job]

                    job_policy_loss = - self.A_coeff * torch.min(surr1, surr2)
                    job_entropy_loss = - self.entropy_coeff_job * jobs_dist_entropy
                    job_actor_loss = job_policy_loss + job_entropy_loss

                    if (job_actor_cat_loss is None) and (job_actor_loss is not None):
                        job_actor_cat_loss = job_actor_loss
                        job_entropy_cat_loss = jobs_dist_entropy
                        job_policy_cat_loss = job_policy_loss
                    elif job_actor_loss is not None:
                        job_actor_cat_loss = torch.cat((job_actor_cat_loss, job_actor_loss))
                        try:
                            job_entropy_cat_loss = torch.cat((job_entropy_cat_loss, jobs_dist_entropy))
                        except:
                            if job_entropy_cat_loss.dim() == 0:
                                job_entropy_cat_loss = job_entropy_cat_loss.unsqueeze(0)
                            if jobs_dist_entropy.dim() == 0:
                                jobs_dist_entropy = jobs_dist_entropy.unsqueeze(0)
                            job_entropy_cat_loss = torch.cat((job_entropy_cat_loss, jobs_dist_entropy))
                        job_policy_cat_loss = torch.cat((job_policy_cat_loss, job_policy_loss))

                job_loss = job_actor_cat_loss.mean()

                if value_loss is not None:
                    critic_loss_epochs += value_loss.mean().detach()
                if job_entropy_cat_loss is not None:
                    job_entropy_loss_epochs += job_entropy_cat_loss.mean().detach()
                if job_policy_cat_loss is not None:
                    job_policy_loss_epochs += job_policy_cat_loss.mean().detach()

                self.optimizer_central.zero_grad()
                value_loss.backward()
                self.optimizer_central.step()

                if (job_actor_cat_loss is not None) and (i % 1 == 0):
                    self.optimizer_job_actor.zero_grad()
                    job_loss.backward()
                    self.optimizer_job_actor.step()

                for name, param in self.policy.named_parameters():
                    if torch.any(torch.isnan(param)):
                        # print(name, param)
                        pass
        current_lr = self.optimizer_central.param_groups[0]['lr']
        print('Learning rate:', current_lr)

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        return 0 / (self.K_epochs * (num_complete_minibatches+r)), \
               0 / (self.K_epochs * (num_complete_minibatches+r)), \
               critic_loss_epochs.item() / (self.K_epochs * (num_complete_minibatches+r)), \
               0/ (self.K_epochs * (num_complete_minibatches+r)), \
               job_entropy_loss_epochs.item() / (self.K_epochs * (num_complete_minibatches+r)), \
               0 / (self.K_epochs * (num_complete_minibatches+r)), \
               job_policy_loss_epochs.item() / (self.K_epochs * (num_complete_minibatches+r)), \
               avg_rewards.item() / (self.num_envs*train_paras["update_freq"])