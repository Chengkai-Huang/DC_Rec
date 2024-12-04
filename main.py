import time
import numpy as np
import pandas as pd
import math
import random
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import os
import time as Time
from utility import pad_history, calculate_hit, extract_axis_1, mean_flat
from collections import Counter
from Modules_ori import *
import copy
import logging
from utils import layers
import pickle
from torch.utils.data import Dataset, DataLoader
from utils import data_helper

logging.getLogger().setLevel(logging.INFO)



class MyDataset(Dataset):
    def __init__(self, data):
        self.all_seq = list(data['seq'].values)
        self.length = list(data['len_seq'].values)
        self.target = list(data['next'].values)

    def __len__(self):
        return len(self.all_seq)

    def __getitem__(self, index):
        seq = self.all_seq[index]
        target = self.target[index]
        len_seq = self.length[index]
        return torch.tensor(seq).to(torch.long), torch.tensor(target).to(torch.long), torch.tensor(len_seq).to(torch.long)

    def _getseq(self, idx):
        return self.all_seqseq[idx]

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='yc',
                        help='yc, ks, zhihu, amazon_beauty, steamm, ml-1m, toys')
    parser.add_argument('--random_seed', type=int, default=100,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', type=int, default=1,
                        help='gru_layers')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--timesteps', type=int, default=500,
                        help='timesteps for diffusion')
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help='beta end of diffusion')
    parser.add_argument('--beta_start', type=float, default=0.0001,
                        help='beta start of diffusion')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--l2_decay', type=float, default=0,
                        help='l2 loss reg coef.')
    parser.add_argument('--cuda', type=int, default=1,
                        help='cuda device.')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--w', type=float, default=4.0,
                        help='dropout ')
    parser.add_argument('--p', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--report_epoch', type=bool, default=True,
                        help='report frequency')
    parser.add_argument('--diffuser_type', type=str, default='mlp1',
                        help='type of diffuser.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='type of optimizer.')
    parser.add_argument('--beta_sche', nargs='?', default='linear',
                        help='')
    parser.add_argument('--descri', type=str, default='',
                        help='description of the work.')
    parser.add_argument('--save_path', type=str, default='./exp', help='Model saving path.')
    parser.add_argument('--save_tag', type=str, default='myrec_', help='Model saving tag.')
    parser.add_argument('--model_path', default="model_path", help='Model saving path')
    parser.add_argument('--mode', choices=["train", "test"], help='Model mode')
    parser.add_argument('--log_file', default='log/', help='log dir path')
    parser.add_argument('--loss_type', default='l2', help='loss type')
    parser.add_argument('--max_len', type=int, default=50, help='seq length')
    parser.add_argument('--num_block', type=int, default=4, help='number of transformer blocks')
    parser.add_argument('--embedding_dropout_ratio', type=int, default=4)
    parser.add_argument('--knowledge_ratio', type=float, default=0.1)
    parser.add_argument('--bad_count', type=int, default=5)
    parser.add_argument('--log', action='store_true', help='print log')


    return parser.parse_args()


args = parse_args()

if args.log:
    if not os.path.exists(args.log_file):
        os.makedirs(args.log_file)
    if not os.path.exists(args.log_file + args.data):
        os.makedirs(args.log_file + args.data)

    logging.basicConfig(level=logging.INFO, filename=args.log_file + args.data + '/' + time.strftime("%Y-%m-%d_%H-%M-%S",
                                                                                                     time.localtime()) + '.log',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s',
                        filemode='w')
    logger = logging.getLogger(__name__)
    logger.info(args)

print(args)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class myDiffusion():
    def __init__(self, timesteps, beta_start, beta_end, w):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.w = w

        if args.beta_sche == 'linear':
            # self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start,
            #                                   beta_end=self.beta_end)

            scale = 1000 / self.timesteps
            self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start * scale,
                                              beta_end=self.beta_end * scale)
        elif args.beta_sche == 'exp':
            self.betas = exp_beta_schedule(timesteps=self.timesteps)
        elif args.beta_sche == 'cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif args.beta_sche == 'sqrt':
            self.betas = torch.tensor(betas_for_alpha_bar(self.timesteps, lambda t: 1 - np.sqrt(t + 0.0001), )).float()
        elif args.beta_sche == 'trunc_lin':
            scale = 1000 / self.timesteps
            # if beta_end > 1:
            #     beta_end = scale * 0.001 + 0.01
            self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start * scale + 0.01,
                                              beta_end=self.beta_end * scale + 0.01)

        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                    1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.adaptive_lmda = 0.2
        self.all_x = list()


    def _scale_timesteps(self, t):
        return t * torch.tensor(1000 / self.timesteps).to(torch.long)

    def q_sample(self, x_start, t, noise=None):
        # print(self.betas)
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t


    def update_lmda(self, epoch=0):
        self.adaptive_lmda = 0.2 * (1-min(epoch, 150)/150) ** 9 + 0.003 # beauty


    def p_losses(self, denoise_model, x_start, h, t, target, noise=None, loss_type="l2", item_embeddings=None, epoch=0):
        #
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_x = denoise_model(x_noisy, h, diff._scale_timesteps(t))


        loss_info = None
        if loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x)
        elif loss_type == 'l2':
            loss = F.mse_loss(x_start, predicted_x)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x)
        elif loss_type == "ce":
            if item_embeddings == None:
                item_embeddings = denoise_model.item_embeddings
            score = torch.matmul(predicted_x, item_embeddings.weight.t())
            loss = F.cross_entropy(score, target)
        elif loss_type == "mix":
            if item_embeddings == None:
                item_embeddings = denoise_model.item_embeddings
            score = torch.matmul(predicted_x[:, -1, :], item_embeddings.weight.t())
            loss_ce = F.cross_entropy(score, target)
            loss_mse = F.mse_loss(x_start[:, -1, :], predicted_x[:, -1, :])

            loss_r = mean_flat(predicted_x ** 2).mean()

            lmda = self.adaptive_lmda
            loss = lmda * loss_ce + (1-lmda) * loss_mse + loss_r
            loss_info = {"ce": loss_ce.item(), "mse": loss_mse.item(), "l2": loss_r.item(), "lmda": lmda}
        elif loss_type == "mse":
            score = torch.matmul(predicted_x, denoise_model.item_embeddings.weight.t())
            target_score = torch.matmul(x_start, denoise_model.item_embeddings.weight.t())
            loss = F.mse_loss(score, target_score)
        else:
            raise NotImplementedError()

        return predicted_x, loss, loss_info

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_xstart_from_noise(self, x_t, t, noise):

        assert x_t.shape == noise.shape
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    @torch.no_grad()
    def p_sample(self, model, x, h, t, t_index):

        x_t = x
        x_start = model(x_t, h, diff._scale_timesteps(t))

        model_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)

            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, h, noise=None):

        if noise is None:
            x = torch.randn_like(h)
        else:
            x = noise

        # self.x_start_wout_target = noise[:, :-1, :] # 26/08/2024

        for n in reversed(range(0, self.timesteps)):
            if n >= 0:
                x = self.p_sample(model, x, h, torch.full((h.shape[0],), n, device=device, dtype=torch.long), n)

        return x[:, -1, :]


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class AttnDiffu(nn.Module):
    def __init__(self, hidden_size, item_num, seq_size, dropout, num_head=4, num_block=4):
        super(AttnDiffu, self).__init__()
        self.seq_size = seq_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = dropout
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        self.time_embeddings = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )
        self.positional_embeddings = nn.Embedding(
            num_embeddings=self.seq_size,
            embedding_dim=hidden_size
        )
        self.embed_dropout = nn.Dropout(self.dropout * args.embedding_dropout_ratio)
        self.final_layer = layers.FinalLayer(hidden_size, dropout)

        self.num_block=num_block
        self.mask = None
        self.extent_mask = None

        self.emb_layernorm = layers.LayerNorm(hidden_size)

        self.transformer_blocks = nn.ModuleList(
            [layers.AdaLNTransformerBlock(self.hidden_size, num_head, self.dropout) for _ in range(self.num_block)])

    def forward(self, emb_x, emb_h, t):
        emb_t = self.time_embeddings(t).unsqueeze(1)
        emb_x = emb_x  
        for transformer in self.transformer_blocks:
            emb_x = transformer.forward(emb_x, emb_h, emb_t, self.mask, self.extent_mask)

        out = self.final_layer(emb_x, emb_t + emb_h)

        return out

    def cacu_x(self, target, history, item_embeddings=None):
        x = torch.cat([history, target.unsqueeze(1)], dim=1)[:,-args.max_len:]

        if item_embeddings is None:
            emb_x = self.item_embeddings(x)
        else:
            emb_x = item_embeddings(x)
        emb_x += self.positional_embeddings(torch.arange(self.seq_size).to(emb_x.device))
        emb_x = self.embed_dropout(emb_x)
        emb_x = self.emb_layernorm(emb_x)
        return emb_x

    def cacu_h(self, h, item_embeddings=None):
        if item_embeddings is None:
            emb_h = self.item_embeddings(h)
        else:
            emb_h = item_embeddings(h)
        emb_h += self.positional_embeddings(torch.arange(self.seq_size).to(emb_h.device))
        emb_h = self.embed_dropout(emb_h)
        emb_h = self.emb_layernorm(emb_h)
        return emb_h


    def cacu_knowledge(self, history, item_embeddings=None):
        noise = torch.zeros_like(history[:, -1])
        noise = torch.cat([history, noise.unsqueeze(1)], dim=1)[:,-args.max_len:]

        if item_embeddings is None:
            noise = self.item_embeddings(noise)
        else:
            noise = item_embeddings(noise)
        noise += self.positional_embeddings(torch.arange(self.seq_size).to(noise.device))
        noise = self.embed_dropout(noise)
        noise = self.emb_layernorm(noise)
        return noise


    def predict(self, history, diff, noise=None, item_embedding=None):
        his_embedding = history
        x = diff.sample(self, his_embedding, noise)

        if item_embedding is None:
            test_item_emb = self.item_embeddings.weight
        else:
            test_item_emb = item_embedding.weight
        scores = torch.matmul(x, test_item_emb.transpose(0, 1))

        return scores

def mytrain(train_data, diff, device):
    model = AttnDiffu(args.hidden_factor, item_num, seq_size, args.dropout_rate, num_block=args.num_block, num_head=4)
    item_embeddings = None

    # model = torch.load(args.model_path)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)


    model.to(device)
    if args.data in ["amazon_beauty", "steam", "ml-1m", "toys"]:
        train_dataloader = train_data.get_pytorch_dataloaders()
    else:
        train_dataset = MyDataset(train_data)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    total_step = 0
    hr_max = 0
    best_epoch = 0

    best_metrics_dict = {'Best_HR@10': 0, 'Best_NDCG@10': 0, 'Best_HR@20': 0, 'Best_NDCG@20': 0, 'Best_HR@50': 0,
                         'Best_NDCG@50': 0}


    bad_count = 0
    for epoch in range(args.epoch):
        start_time = Time.time()
        loss_info = None

        diff.update_lmda(epoch)

        for seq, target, mask, extent_mask in train_dataloader:
            optimizer.zero_grad()
            model.train()

            seq = seq.to(device)
            target = target.to(device).squeeze()
            model.mask = mask.to(device)
            model.extent_mask = extent_mask.to(device)
            x_start = model.cacu_x(target, seq)
            h = model.cacu_h(seq)
            noise = torch.randn_like(x_start) + model.cacu_knowledge(seq) * args.knowledge_ratio

            t = torch.randint(0, args.timesteps, (seq.shape[0],), device=device).long()

            predicted_x, loss, loss_info = diff.p_losses(model, x_start, h, t, target, noise=noise, loss_type=args.loss_type, item_embeddings=item_embeddings, epoch=epoch)

            loss.backward()
            optimizer.step()


        # scheduler.step()
        if args.report_epoch:
            flag_update = 0
            if epoch % 1 == 0:
                if loss_info is None:
                    print("Epoch {:03d}; ".format(epoch) + 'Train loss: {:.4f}; '.format(
                        loss) + "Time cost: " + Time.strftime(
                        "%H: %M: %S", Time.gmtime(Time.time() - start_time)))
                else:
                    print("Epoch {:03d}; ".format(epoch) + 'Train loss: {:.4f}; '.format(
                        loss) + 'CE loss: {:.4f}; '.format(loss_info["ce"]) + 'MSE loss: {:.4f}; '.format(
                        loss_info["mse"]) + 'L2 loss: {:.4f}; '.format(
                        loss_info["l2"]) + 'Loss lmda: {:.4f}; '.format(
                        loss_info["lmda"]) + "Time cost: " + Time.strftime(
                        "%H: %M: %S", Time.gmtime(Time.time() - start_time)))

            if (epoch + 1) % 5 == 0:

                eval_start = Time.time()
                print('-------------------------- VAL PHRASE --------------------------')
              
                metrics_dict = evaluate(model, eval_data, diff, device)
                print("Evalution cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time() - eval_start)))
                print('----------------------------------------------------------------')

                if args.log:
                    log_dict = {"epoch":epoch}
                    log_dict.update(metrics_dict)
                    logger.info(log_dict)

                for key_temp, values_temp in metrics_dict.items():
                    values = round(values_temp * 100, 4)
                    if values > best_metrics_dict['Best_' + key_temp]:
                        flag_update = 1
                        bad_count = 0
                        best_metrics_dict['Best_' + key_temp] = values

                if flag_update == 0:
                    bad_count += 1
                else:
                    best_model = copy.deepcopy(model)
                    torch.save(best_model, os.path.join(args.save_path, args.save_tag + args.data + '.pth'))
                if bad_count >= args.bad_count:
                    break

        else:
            best_model = copy.deepcopy(model)
            torch.save(best_model, os.path.join(args.save_path, args.save_tag + args.data + '.pth'))

    print('-------------------------- TEST PHRASE -------------------------')
  
    test_metrics_dict = evaluate(best_model, test_data, diff, device)

    if args.log:
        logger.info(test_metrics_dict)

def evaluate(model, eval_data, diff, device, pre_model=None):
    model.eval()

    batch_size = args.batch_size
    evaluated = 0
    total_clicks = 1.0
    total_purchase = 0.0
    total_reward = [0, 0, 0, 0]
    hit_clicks = [0, 0, 0, 0]
    ndcg_clicks = [0, 0, 0, 0]
    hit_purchase = [0, 0, 0, 0]
    ndcg_purchase = [0, 0, 0, 0]

    if args.data in ["amazon_beauty", "steam", "ml-1m", "toys"]:
        eval_dataloader = eval_data.get_pytorch_dataloaders()
    else:
        eval_dataset = MyDataset(eval_data)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


    losses = []
    for seq, target, mask, extent_mask in eval_dataloader:
        seq = seq.to(device)
        target_np = target.squeeze().numpy().tolist()

        with torch.no_grad():
            model.extent_mask = extent_mask.to(device)
            model.mask = mask.to(device)
            item_embeddings = None
            his_embedding = model.cacu_h(seq)
            noise = torch.randn_like(his_embedding) + model.cacu_knowledge(seq) * args.knowledge_ratio
            score = model.predict(his_embedding, diff, noise, item_embedding=item_embeddings)

        _, topK = score.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2 = np.flip(topK, axis=1)
        sorted_list2 = sorted_list2
        calculate_hit(sorted_list2, topk, target_np, hit_purchase, ndcg_purchase)

        total_purchase += batch_size

    hr_list = []
    ndcg_list = []

    print('{:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@' + str(topk[0]), 'NDCG@' + str(topk[0]),
                                                                   'HR@' + str(topk[1]), 'NDCG@' + str(topk[1]),
                                                                   'HR@' + str(topk[2]), 'NDCG@' + str(topk[2])))
    for i in range(len(topk)):
        hr_purchase = hit_purchase[i] / total_purchase
        ng_purchase = ndcg_purchase[i] / total_purchase

        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase[0, 0])

        # if i == 1:
        #     hr_20 = hr_purchase

    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], (ndcg_list[0]), hr_list[1],
                                                                               (ndcg_list[1]), hr_list[2],
                                                                               (ndcg_list[2])))

    # losses = torch.stack(losses, dim=0).mean()
    # print("Loss: {:<10.3f}".format(losses))

    metrics_dict = {'HR@10': hr_list[0], 'NDCG@10': ndcg_list[0], 'HR@20': hr_list[1], 'NDCG@20': ndcg_list[1],
                    'HR@50': hr_list[2], 'NDCG@50': ndcg_list[2]}

    return metrics_dict


if __name__ == '__main__':

    # args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    setup_seed(args.random_seed)

    topk = [5, 10, 20]


    #personal test
    if args.data in ["amazon_beauty", "steam", "ml-1m", "toys"]:
        path_data = f'./data/beauty/dataset.pkl'
        seq_size = args.max_len
        with open(path_data, 'rb') as f:
            data_raw = pickle.load(f)
        item_num = len(data_raw['smap'])
        train_data = data_helper.Data_Train(data_raw['train'], args)
        eval_data = data_helper.Data_Val(data_raw['train'], data_raw['val'], args)
        test_data = data_helper.Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args)
    else:
        data_directory = './data/' + args.data
        data_statis = pd.read_pickle(
            os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing seq_size and item_num
        seq_size = data_statis['seq_size'][0]  # the length of history to define the seq
        item_num = data_statis['item_num'][0]  # total number of items

        train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
        eval_data = pd.read_pickle(os.path.join(data_directory, 'val_data.df'))
        test_data = pd.read_pickle(os.path.join(data_directory, 'test_data.df'))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timesteps = args.timesteps

    diff = myDiffusion(args.timesteps, args.beta_start, args.beta_end, args.w)

    if args.mode == "train":
        mytrain(train_data, diff, device)

    elif args.mode == "test":
        t1 = time.time()
        model = torch.load(args.model_path)
        pre_model = None
        metrics = evaluate(model, test_data, diff, device, pre_model)
        print("Time Cost:", time.time() - t1)

    else:
        raise NotImplementedError("Unrecognized mode {}".format(args.mode))