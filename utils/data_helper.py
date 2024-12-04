import torch.utils.data as data_utils
import torch


class TrainDataset(data_utils.Dataset):
    def __init__(self, id2seq, max_len):
        self.id2seq = id2seq
        self.max_len = max_len

    def __len__(self):
        return len(self.id2seq)

    def __getitem__(self, index):
        seq = self._getseq(index)
        labels = [seq[-1]]
        tokens = seq[:-1]
        tokens = tokens[-self.max_len:]
        len_token = len(tokens)
        len_mask = self.max_len - len_token

        tokens = [0] * len_mask + tokens
        extent_len_token = min(self.max_len, len_token+1)
        mask = [0.0] * len_mask + [1.0] * len_token
        extent_mask = ([0.0] * len_mask + [1.0] * extent_len_token)[-self.max_len:]

        # tokens = tokens + [0] * mask_len
        return torch.LongTensor(tokens), torch.LongTensor(labels), torch.tensor(mask), torch.tensor(extent_mask)

    def _getseq(self, idx):
        return self.id2seq[idx]


class Data_Train():
    def __init__(self, data_train, args):
        self.u2seq = data_train
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        self.split_onebyone()

    def split_onebyone(self):
        self.id_seq = {}
        self.id_seq_user = {}
        idx = 0
        if type(self.u2seq) is list:  # yelp and sports dataset
            for i, seq_temp in enumerate(self.u2seq):
                for star in range(len(seq_temp) - 1):
                    self.id_seq[idx] = seq_temp[:star + 2]
                    idx += 1
        else:
            for user_temp, seq_temp in self.u2seq.items():
                for star in range(len(seq_temp) - 1):
                    self.id_seq[idx] = seq_temp[:star + 2]
                    self.id_seq_user[idx] = user_temp
                    idx += 1
                # self.id_seq[idx] = seq_temp
                # idx += 1

    def get_pytorch_dataloaders(self):
        dataset = TrainDataset(self.id_seq, self.max_len)
        return data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)


class ValDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys()) if type(self.u2seq) is dict else None # yelp and sports dataset
        self.u2answer = u2answer
        self.max_len = max_len

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, index):
        if self.users is None:  # yelp and sports dataset
            seq = self.u2seq[index]
            answer = [self.u2answer[index]]
        else:
            user = self.users[index]
            seq = self.u2seq[user]
            answer = self.u2answer[user]
        seq = seq[-self.max_len:]
        len_seq = len(seq)
        len_padding = self.max_len - len_seq

        seq = [0] * len_padding + seq
        extent_len_seq = min(self.max_len, len_seq + 1)
        padding = [0.0] * len_padding + [1.0] * len_seq
        extent_padding = ([0.0] * len_padding + [1.0] * extent_len_seq)[-self.max_len:]

        # seq = seq + [0] * padding_len
        return torch.LongTensor(seq), torch.LongTensor(answer), torch.tensor(padding), torch.tensor(extent_padding)


class Data_Val():
    def __init__(self, data_train, data_val, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2answer = data_val
        self.max_len = args.max_len

    def get_pytorch_dataloaders(self):
        dataset = ValDataset(self.u2seq, self.u2answer, self.max_len)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader


class TestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2_seq_add, u2answer, max_len):
        self.u2seq = u2seq
        self.u2seq_add = u2_seq_add
        self.users = sorted(self.u2seq.keys()) if type(self.u2seq) is dict else None # yelp and sports dataset
        self.u2answer = u2answer
        self.max_len = max_len

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, index):
        if self.users == None:  # yelp and sports dataset
            seq = self.u2seq[index] + [self.u2seq_add[index]] if type(self.u2seq_add[index]) is int else self.u2seq_add[index]
            answer = [self.u2answer[index]]
        else:
            user = self.users[index]
            seq = self.u2seq[user] + self.u2seq_add[user]
            # seq = self.u2seq[user]
            answer = self.u2answer[user]
        seq = seq[-self.max_len:]
        len_seq = len(seq)
        len_padding = self.max_len - len_seq

        seq = [0] * len_padding + seq
        extent_len_seq = min(self.max_len, len_seq + 1)
        padding = [0.0] * len_padding + [1.0] * len_seq
        extent_padding = ([0.0] * len_padding + [1.0] * extent_len_seq)[-self.max_len:]
        # seq = seq + [0] * padding_len
        return torch.LongTensor(seq), torch.LongTensor(answer), torch.tensor(padding), torch.tensor(extent_padding)



class Data_Test():
    def __init__(self, data_train, data_val, data_test, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2seq_add = data_val
        self.u2answer = data_test
        self.max_len = args.max_len

    def get_pytorch_dataloaders(self):
        dataset = TestDataset(self.u2seq, self.u2seq_add, self.u2answer, self.max_len)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader


class CHLSDataset(data_utils.Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_temp = self.data[index]
        seq = data_temp[:-1]
        answer = [data_temp[-1]]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        # seq = [0] * padding_len + seq
        seq = seq + [0] * padding_len
        return torch.LongTensor(seq), torch.LongTensor(answer)


class Data_CHLS():
    def __init__(self, data, args):
        self.batch_size = args.batch_size
        self.max_len = args.max_len
        self.data = data

    def get_pytorch_dataloaders(self):
        dataset = CHLSDataset(self.data, self.max_len)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader