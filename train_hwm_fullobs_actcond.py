import sys
import os
import logging
from datetime import datetime
import numpy as np
import torch
import torch as th
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
# from hssm import EnvModel
from utils import preprocess, postprocess, full_dataloader, log_train, log_test, plot_rec, plot_gen, concat, gumbel_sampling, log_density_concrete
from modules import * # for the HSSM modules
LOGGER = logging.getLogger(__name__)

# PLotting related
import matplotlib.pyplot as plt
import seaborn as sns

# dataset related
import gym
import d4rl
import d4rl_atari

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Hierarchical World Model argument parser")
    parser.add_argument('--seed', type=int, default=111)

    # data size
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seq-size', type=int, default=80) # THe size of the sequence the model is trained upon, not the whole sequence size it seems.
    parser.add_argument('--init-size', type=int, default=5)

    # model hyparms
    parser.add_argument('--temporal-state-size', type=int, default=8) # c's dim
    parser.add_argument('--temporal-belief-size', type=int, default=128) # z's dim
    parser.add_argument('--state-belief-size', type=int, default=128) # h's dim
    parser.add_argument('--num-layers', type=int, default=3)
    
    # state distribution
    # TODO: consider making this learned too
    parser.add_argument('--state-std', type=float, default=1.0) # Defautl is 1.0

    # optimization
    parser.add_argument('--learn-rate', type=float, default=0.0005)
    parser.add_argument('--grad-clip', type=float, default=10.0)
    parser.add_argument('--max-iters', type=int, default=500000)

    # subsequence prior params
    parser.add_argument('--seg-num', type=int, default=10)
    parser.add_argument('--seg-len', type=int, default=20)

    # gumbel params
    parser.add_argument('--max-beta', type=float, default=1.0)
    parser.add_argument('--min-beta', type=float, default=0.1)
    parser.add_argument('--beta-anneal', type=float, default=100)

    # log dir
    parser.add_argument('--log-dir', type=str, default='./asset/log/')
    return parser.parse_args()

def set_exp_name(args):
    exp_name = 'hwd_umaze_fullobs_actcond'
    exp_name += '_b{}'.format(args.batch_size)
    exp_name += '_l{}_i{}'.format(args.seq_size, args.init_size)
    exp_name += '_b{}_s{}_c{}'.format(args.temporal_belief_size, args.temporal_state_size, args.num_layers)
    exp_name += '_gc{}_lr{}'.format(args.grad_clip, args.learn_rate)
    exp_name += '_sg{}-{}'.format(args.seg_num, args.seg_len)
    exp_name += '_gum{}-{}-{}'.format(args.min_beta, args.max_beta, args.beta_anneal)
    exp_name += '_seed{}'.format(args.seed)
    exp_name += '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    
    return exp_name

# parse arguments
args = parse_args()

## some fiddling

# fix seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

# set logger
log_format = '[%(asctime)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stderr)

# set size
seq_size = args.seq_size
init_size = args.init_size

# set device as gpu
device = torch.device('cuda', 0)

from torch.utils.data import Dataset, DataLoader

# TODO: figure more dynamic dataset generation
class MazeDataset(Dataset):
    def __init__(self, length, partition, dataset):
        self.partition = partition
        num_seqs = int(dataset['observations'].shape[0] * 0.8)
        state_shape = dataset["observations"].shape[-1]
        action_shape = dataset['actions'].shape[-1]
        if self.partition == 'train':
            self.state = np.concatenate([dataset['observations'][:num_seqs], dataset['actions'][:num_seqs]], axis=1)
        else:
            self.state = np.concatenate([dataset['observations'][num_seqs:], dataset['actions'][num_seqs:]], axis=1)
        
        self.state = self.state.reshape(-1, 100, state_shape + action_shape)
        self.length = length
        self.full_length = self.state.shape[1]

    def __len__(self):
        return self.state.shape[0]

    def __getitem__(self, index):
        idx0 = np.random.randint(0, self.full_length - self.length)
        idx1 = idx0 + self.length
        state = self.state[index, idx0:idx1].astype(np.float32)
        return state

def full_dataloader(seq_size, init_size, batch_size, test_size=16, dataset=None):
    assert dataset is not None
    train_loader = MazeDataset(length=seq_size + init_size * 2, partition='train', dataset=dataset)
    test_loader = MazeDataset(length=seq_size + init_size * 2, partition='test', dataset=dataset)
    train_loader = DataLoader(dataset=train_loader, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_loader, batch_size=test_size, shuffle=False)
    return train_loader, test_loader

# load dataset
# loading dataset
env = gym.make('maze2d-umaze-v1')
dataset = env.get_dataset()
state_size = env.observation_space.shape[-1]
action_size = env.action_space.shape[-1]
train_loader, test_loader = full_dataloader(seq_size, init_size, args.batch_size, dataset=dataset)

# set writer
exp_name = set_exp_name(args)
writer = SummaryWriter(args.log_dir + exp_name)
LOGGER.info('EXP NAME: ' + exp_name)

class EnvModel(nn.Module):
    def __init__(self,
                 state_size, # fully observable state's dimension (s)
                 action_size, # action's dimension (a)
                 temporal_belief_size, # temporal state belief's dimension (c)
                 temporal_state_size, # temporal state's dimension (z)
                 state_belief_size, # state belief's dimension (h)
                 num_layers, max_seg_len, max_seg_num):
        super(EnvModel, self).__init__()
        
        ################
        # network size #
        ################
        self.state_size = state_size
        self.action_size = action_size
        self.temporal_belief_size = temporal_belief_size
        self.temporal_state_size = temporal_state_size
        self.state_belief_size = state_belief_size
        self.num_layers = num_layers
        self.max_seg_len = max_seg_len
        self.max_seg_num = max_seg_num
    
        ###############
        # init models #
        ###############
        # state space model
        self.state_model = HierarchicalStateSpaceModel(state_size=self.state_size,
                                                       action_size=self.action_size,
                                                       temporal_belief_size=self.temporal_belief_size,
                                                       temporal_state_size=self.temporal_state_size,
                                                       state_belief_size=self.state_belief_size,
                                                       num_layers=self.num_layers,
                                                       max_seg_len=self.max_seg_len,
                                                       max_seg_num=self.max_seg_num)
    def jumpy_generation(self, init_obs_list, seq_size):
        return self.state_model.jumpy_generation(init_obs_list, seq_size)

    def full_generation(self, init_obs_list, seq_size):
        return self.state_model.full_generation(init_obs_list, seq_size)
    
    def semi_online_full_gen(self, trajectory, init_seq_length):
        return self.state_model.semi_online_full_gen(trajectory, init_seq_length)
    
    def forward(self, obs_data_list, seq_size, init_size, obs_std=1.0):
        ############################
        # (1) run over state model #
        ############################
        ### TODO: harmiznize the names with the one in the Hierarchical State Model
        [state_rec_list,
         prior_boundary_log_density_list,
         post_boundary_log_density_list,
         prior_abs_state_list,
         post_abs_state_list,
         prior_obs_state_list,
         post_obs_state_list,
         boundary_data_list,
         prior_boundary_list,
         post_boundary_list] = self.state_model(obs_data_list, seq_size, init_size)
        
        ########################################################
        # (2) compute state_cost (sum over spatial and channels) #
        ########################################################
        obs_target_list = obs_data_list[:, init_size:-init_size]
        state_cost = - Normal(state_rec_list, obs_std).log_prob(obs_target_list[:, :, :self.state_size])
        # Slice at the end separates states from cat(states, actions)
        state_cost = state_cost.sum(dim=[1,2])
        
        #######################
        # (3) compute kl_cost #
        #######################
        # compute kl related to states
        kl_abs_state_list = []
        kl_obs_state_list = []
        for t in range(seq_size):
            # read flag
            read_data = boundary_data_list[:, t].detach()

            # kl divergences (sum over dimension)
            kl_abs_state = kl_divergence(post_abs_state_list[t], prior_abs_state_list[t]) * read_data
            kl_obs_state = kl_divergence(post_obs_state_list[t], prior_obs_state_list[t])
            kl_abs_state_list.append(kl_abs_state.sum(-1))
            kl_obs_state_list.append(kl_obs_state.sum(-1))
        kl_abs_state_list = torch.stack(kl_abs_state_list, dim=1)
        kl_obs_state_list = torch.stack(kl_obs_state_list, dim=1)

        # compute kl related to boundary
        kl_mask_list = (post_boundary_log_density_list - prior_boundary_log_density_list)
        
        # return
        return {'rec_data': state_rec_list,
                'mask_data': boundary_data_list,
                'state_cost': state_cost,
                'kl_abs_state': kl_abs_state_list,
                'kl_state': kl_obs_state_list,
                'kl_mask': kl_mask_list,
                'p_mask': prior_boundary_list.mean,
                'q_mask': post_boundary_list.mean,
                'p_ent': prior_boundary_list.entropy(),
                'q_ent': post_boundary_list.entropy(),
                'beta': self.state_model.mask_beta,
                'train_loss': state_cost.mean() + kl_abs_state_list.mean() + kl_obs_state_list.mean() + kl_mask_list.mean()}

class HierarchicalStateSpaceModel(nn.Module):
    def __init__(self,
                 state_size, # fully observable state's dimension (s)
                 action_size, # action's dimension (a)
                 temporal_belief_size, # temporal state belief's dimension (c)
                 temporal_state_size, # temporal state's dimension (z)
                 state_belief_size, # state belief's dimension (h)
                 num_layers,
                 max_seg_len,
                 max_seg_num):
        super(HierarchicalStateSpaceModel, self).__init__()
        
        ### Network sizes ###
        # Temporal abstraction
        self.temporal_belief_size = temporal_belief_size
        self.temporal_state_size = temporal_state_size
        self.temporal_feat_size = temporal_belief_size # after concating: c || z, determines the dimension of the latter's embedding
        
        # State (dynamics of the RL system)
        self.state_belief_size = state_belief_size
        self.state_feat_size = state_belief_size
        self.state_size = state_size
        self.action_size = action_size
        
        # Other network hyparams
        self.num_layers = num_layers
        self.feat_size = temporal_belief_size # TODO: add a separate parameterization ? 
        
        # sub-sequence information (prior over m_t)
        self.max_seg_len = max_seg_len
        self.max_seg_num = max_seg_num
        
        # for concrete distribution
        self.mask_beta = 1.0
        
        ### Boundary detector ###
        self.prior_boundary = PriorBoundaryDetector(input_size=self.state_size + self.action_size) # hat_p(m_t |S,A) # TODO: add MLP within and add some layers ?
        self.post_boundary = PostBoundaryDetector(input_size=self.state_size + self.action_size, num_layers=self.num_layers) # q(m_t |S,A) # TODO: make this MLP ?
        
        ### Feature extractors ###
        self.abs_feat = LinearLayer(input_size=self.temporal_belief_size + self.temporal_state_size,
                                            output_size=self.temporal_feat_size, nonlinear=nn.Identity()) # maps concat(c,z) -> R^{temporal_feat_size}
        # TODO: Consider additional embedding from S x A -> R to get a more expressive reprensation of fully observable state-action space ?
        
        ### belief initialization ### 
        self.init_abs_belief = nn.Identity()
        self.init_obs_belief = nn.Identity()
        
        ### Beliefs update ###
        self.update_abs_belief = RecurrentLayer(input_size=self.temporal_state_size, hidden_size=self.temporal_belief_size)
        self.update_state_belief = RecurrentLayer(input_size=self.state_size + self.action_size, hidden_size=self.state_belief_size)
        
        ### Posterior encoder ###
        self.abs_post_fwd = RecurrentLayer(input_size=self.state_size + self.action_size, hidden_size=self.temporal_belief_size) # \psi^{fwd}_t
        self.abs_post_bwd = RecurrentLayer(input_size=self.state_size + self.action_size, hidden_size=self.temporal_belief_size) # \psi^{bwd}_t
        self.state_post_fwd = RecurrentLayer(input_size=self.state_size + self.action_size, hidden_size=self.state_belief_size) # \phi^{fwd}_t
        
        ### Prior over temporally abstract state and "real" state ###
        # TODO: rename latent_size to ouput_size for less confusion ?
        # TODO: more RL like LatentDistribution ? namely the std parameterization ?
        self.prior_abs_state = LatentDistribution(input_size=self.temporal_belief_size, latent_size=self.temporal_state_size, feat_size=self.temporal_belief_size) # hat_p(z_t |c_t)
        self.prior_state = LatentDistribution(input_size=self.state_belief_size, latent_size=self.state_size, feat_size=self.state_belief_size) # hat_p(s_t | h_t) ?
        # TODO: consider embedding of sequence of action instead, so as to match p(s_t | s_{t-1}, a_{t-1}), but with action accumulated across the past ?
        
        ### Posterior over temporally abstract state and "real" state ###
        self.post_abs_state = LatentDistribution(input_size=self.temporal_belief_size * 2, latent_size=self.temporal_state_size, feat_size=self.temporal_belief_size) # q(z_t |\psi^{fwd}_t,\psi^{bwd}_t)
        self.post_state = LatentDistribution(input_size=self.temporal_feat_size + self.state_belief_size, latent_size=self.state_size, feat_size=self.state_belief_size) # q(s_t | f(c_t || z_t), \phi^{fwd}_t)
    
    # sampler
    def boundary_sampler(self, log_alpha):
        # sample and return corresponding logit
        if self.training:
            log_sample_alpha = gumbel_sampling(log_alpha=log_alpha, temp=self.mask_beta)
        else:
            log_sample_alpha = log_alpha / self.mask_beta

        # probability
        log_sample_alpha = log_sample_alpha - torch.logsumexp(log_sample_alpha, dim=-1, keepdim=True)
        sample_prob = log_sample_alpha.exp()
        sample_data = torch.eye(2, dtype=log_alpha.dtype, device=log_alpha.device)[torch.max(sample_prob, dim=-1)[1]]

        # sample with rounding and st-estimator
        sample_data = sample_data.detach() + (sample_prob - sample_prob.detach())

        # return sample data and logit
        return sample_data, log_sample_alpha

    # set prior boundary prob
    def regularize_prior_boundary(self, log_alpha_list, boundary_data_list):
        # only for training
        if not self.training:
            return log_alpha_list

        #################
        # sequence size #
        #################
        num_samples = boundary_data_list.size(0)
        seq_len = boundary_data_list.size(1)

        ###################
        # init seg static #
        ###################
        seg_num = log_alpha_list.new_zeros(num_samples, 1)
        seg_len = log_alpha_list.new_zeros(num_samples, 1)

        #######################
        # get min / max logit #
        #######################
        one_prob = 1 - 1e-3
        max_scale = np.log(one_prob / (1 - one_prob))

        near_read_data = log_alpha_list.new_ones(num_samples, 2) * max_scale
        near_read_data[:, 1] = - near_read_data[:, 1]
        near_copy_data = log_alpha_list.new_ones(num_samples, 2) * max_scale
        near_copy_data[:, 0] = - near_copy_data[:, 0]

        # for each step
        new_log_alpha_list = []
        for t in range(seq_len):
            ##########################
            # (0) get length / count #
            ##########################
            read_data = boundary_data_list[:, t, 0].unsqueeze(-1)
            copy_data = boundary_data_list[:, t, 1].unsqueeze(-1)
            seg_len = read_data * 1.0 + copy_data * (seg_len + 1.0)
            seg_num = read_data * (seg_num + 1.0) + copy_data * seg_num
            over_len = torch.ge(seg_len, self.max_seg_len).float().detach()
            over_num = torch.ge(seg_num, self.max_seg_num).float().detach()

            ############################
            # (1) regularize log_alpha #
            ############################
            # if read enough times (enough segments), stop
            new_log_alpha = over_num * near_copy_data + (1.0 - over_num) * log_alpha_list[:, t]

            # if length is too long (long segment), read
            new_log_alpha = over_len * near_read_data + (1.0 - over_len) * new_log_alpha

            ############
            # (2) save #
            ############
            new_log_alpha_list.append(new_log_alpha)

        # return
        return torch.stack(new_log_alpha_list, dim=1)
    
    # forward for reconstruction
    def forward(self, obs_data_list, seq_size, init_size):
        ### data size ###
        num_samples = obs_data_list.size(0)
        full_seq_size = obs_data_list.size(1)
        
        ### boundary sampling ###
        post_boundary_log_alpha_list = self.post_boundary(obs_data_list)
        boundary_data_list, post_boundary_sample_logit_list = self.boundary_sampler(post_boundary_log_alpha_list)
        
        boundary_data_list[:, :(init_size + 1), 0] = 1.0 # these two lines set the INIT_SIZE for the first elements
        boundary_data_list[:, :(init_size + 1), 1] = 0.0
        boundary_data_list[:, -init_size:, 0] = 1.0 # these two lines set the INIT_SIZE for the last elements
        boundary_data_list[:, -init_size:, 1] = 0.0
        
        ### Posterior encoding ###
        abs_post_fwd_list = []
        abs_post_bwd_list = []
        state_post_fwd_list = []
        abs_post_fwd = obs_data_list.new_zeros(num_samples, self.temporal_belief_size)
        abs_post_bwd = obs_data_list.new_zeros(num_samples, self.temporal_belief_size)
        state_post_fwd = obs_data_list.new_zeros(num_samples, self.state_belief_size)
        for fwd_t, bwd_t in zip(range(full_seq_size), reversed(range(full_seq_size))):
            # forward encoding
            fwd_copy_data = boundary_data_list[:, fwd_t, 1].unsqueeze(-1)
            abs_post_fwd = self.abs_post_fwd(obs_data_list[:, fwd_t], abs_post_fwd)
            state_post_fwd = self.state_post_fwd(obs_data_list[:, fwd_t], state_post_fwd)
            abs_post_fwd_list.append(abs_post_fwd)
            state_post_fwd_list.append(state_post_fwd)
            
            # backward encoding
            bwd_copy_data = boundary_data_list[:, bwd_t, 1].unsqueeze(-1)
            abs_post_bwd = self.abs_post_bwd(obs_data_list[:, bwd_t], abs_post_bwd)
            abs_post_bwd_list.append(abs_post_bwd)
            abs_post_bwd = bwd_copy_data * abs_post_bwd
        abs_post_bwd_list = abs_post_bwd_list[::-1]
    
        ### Init lists ###
        state_rec_list = []
        prior_abs_state_list = []
        post_abs_state_list = []
        prior_obs_state_list = []
        post_obs_state_list = []
        prior_boundary_log_alpha_list = []
        
        ### Init latent and "real" state ###
        abs_belief = obs_data_list.new_zeros(num_samples, self.temporal_belief_size)
        abs_state = obs_data_list.new_zeros(num_samples, self.temporal_state_size)
        state_belief = obs_data_list.new_zeros(num_samples, self.state_belief_size)
        state = obs_data_list.new_zeros(num_samples, self.state_size)
        
        ### Forward transition ###
        for t in range(init_size, init_size + seq_size):
            ## (0) get the mask data ##
            read_data = boundary_data_list[:, t, 0].unsqueeze(-1)
            copy_data = boundary_data_list[:, t, 1].unsqueeze(-1)
            
            ## (1) sample abstract state ##
            if t == init_size:
                abs_belief = self.init_abs_belief(abs_post_fwd_list[t - 1])
            else:
                abs_belief = read_data * self.update_abs_belief(abs_state, abs_belief) + copy_data * abs_belief
            prior_abs_state = self.prior_abs_state(abs_belief) # hat_p(z_t | c_t)
            post_abs_state = self.post_abs_state(concat(abs_post_fwd_list[t - 1], abs_post_bwd_list[t])) # q( z_t | \psi^fwd_t, \psi^bwd_t)
            abs_state = read_data * post_abs_state.rsample() + copy_data * abs_state
            abs_feat = self.abs_feat(concat(abs_belief, abs_state)) # f(c_t || z_t)
            
            ## (2) sample "real" state ##
            state_belief = self.update_state_belief(obs_data_list[:, t], state_belief) # h = f(s,a, h), as in the standard world models
            prior_state = self.prior_state(state_belief) # hat_p(s_t | h_t)
            post_state = self.post_state(concat(state_post_fwd_list[t], abs_feat))
            rec_state = post_state.rsample()
            
            ### (3) appending reconstructed state ###
            state_rec_list.append(rec_state)
            
            ### (4) mask prior ###
            rec_state_action = th.cat([rec_state, obs_data_list[:, t, self.state_size:]], 1) # trick to get s_t, a_t to sample hat_p(m_t | s_t, a_t)
            prior_boundary_log_alpha = self.prior_boundary(rec_state_action)
            
            ### (5) append other intermetdiate data ###
            prior_boundary_log_alpha_list.append(prior_boundary_log_alpha)
            prior_abs_state_list.append(prior_abs_state)
            post_abs_state_list.append(post_abs_state)
            prior_obs_state_list.append(prior_state)
            post_obs_state_list.append(post_state)
        
        # stacking results
        state_rec_list = torch.stack(state_rec_list, dim=1)
        prior_boundary_log_alpha_list = torch.stack(prior_boundary_log_alpha_list, dim=1)
        
        # remove padding
        boundary_data_list = boundary_data_list[:, init_size:(init_size + seq_size)]
        post_boundary_log_alpha_list = post_boundary_log_alpha_list[:, (init_size + 1):(init_size + 1 + seq_size)]
        post_boundary_sample_logit_list = post_boundary_sample_logit_list[:, (init_size + 1):(init_size + 1 + seq_size)]
        
        # fix prior by constraints
        prior_boundary_log_alpha_list = self.regularize_prior_boundary(prior_boundary_log_alpha_list, boundary_data_list)
        
        # compute log-density
        prior_boundary_log_density = log_density_concrete(prior_boundary_log_alpha_list,
                                                          post_boundary_sample_logit_list,
                                                          self.mask_beta)
        post_boundary_log_density = log_density_concrete(post_boundary_log_alpha_list,
                                                         post_boundary_sample_logit_list,
                                                         self.mask_beta)
        
        # compute boundary probability
        prior_boundary_list = F.softmax(prior_boundary_log_alpha_list / self.mask_beta, -1)[..., 0]
        post_boundary_list = F.softmax(post_boundary_log_alpha_list / self.mask_beta, -1)[..., 0]
        prior_boundary_list = Bernoulli(probs=prior_boundary_list)
        post_boundary_list = Bernoulli(probs=post_boundary_list)
        boundary_data_list = boundary_data_list[..., 0].unsqueeze(-1)
        
        # return
        return [state_rec_list,
                prior_boundary_log_density,
                post_boundary_log_density,
                prior_abs_state_list,
                post_abs_state_list,
                prior_obs_state_list,
                post_obs_state_list,
                boundary_data_list,
                prior_boundary_list,
                post_boundary_list]
    
    # generation forward
    def full_generation(self, init_data_list, seq_size):
        # placeholder
        return [], [], []

    def semi_online_full_gen(self, trajectory, init_seq_length=1):
        # By default we assume the initialization size is 1, to match with the RL model
        return [], [], []
    
    # generation forward
    def jumpy_generation(self, init_data_list, seq_size):
        return [], [], []

# training
model = EnvModel(state_size, action_size, temporal_belief_size=args.temporal_belief_size, temporal_state_size=args.temporal_state_size,
                 state_belief_size=args.state_belief_size, num_layers=args.num_layers, max_seg_len=args.seg_len, max_seg_num=args.seg_num).to(device)
optimizer = Adam(params=model.parameters(), lr=args.learn_rate, amsgrad=True)

pre_test_full_data_list = iter(test_loader).next()

# for each iter
b_idx = 0
while b_idx <= args.max_iters:
    # for each batch
    for train_obs_list in train_loader:
        b_idx += 1
        # mask temp annealing
        if args.beta_anneal:
            model.state_model.mask_beta = (args.max_beta - args.min_beta) * 0.999 ** (b_idx / args.beta_anneal) + args.min_beta
        else:
            model.state_model.mask_beta = args.max_beta

        ##############
        # train time #
        ##############
         # run model with train mode
        model.train()
        optimizer.zero_grad()
        results = model(train_obs_list.to(device), seq_size, init_size, args.state_std)
        
        # get train loss and backward update
        train_total_loss = results['train_loss']
        train_total_loss.backward()
        if args.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        
        # log
        if b_idx % 100 == 0:
            log_str, log_data = log_train(results, writer, b_idx)
            LOGGER.info(log_str, *log_data)
        
        # model saving
        if b_idx % 1000 == 0:
            savepath = os.path.join(args.log_dir, exp_name)
            savepath = os.path.join(savepath, "models")
            if not os.path.exists(savepath):
                os.makedirs(savepath)
                
            savepath = os.path.join(savepath, "env_model_%d" % b_idx)
            torch.save(model, savepath)