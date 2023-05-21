import torch
import utils


class Emb():
    def __init__(self, episode_length=10, emb_dim=30):
        self.episode_length = episode_length
        self.emb_dim = emb_dim

        ietm_path = '../data/processed_data/MI/item'
        object = utils.pickle_load(ietm_path)
        self.embeddings = torch.from_numpy(object['item'])

    def get_seq_emb(self, seq):
        assert len(seq) <= self.episode_length
        seq = torch.LongTensor(seq)
        ses = []
        for s in seq:
            se = self.embeddings[s]
            ses.append(se)

        L, l = self.episode_length, len(ses)
        E = self.emb_dim
        pad = torch.zeros((L - l, E)).long()
        ses = torch.tensor([item.detach().numpy() for item in ses])
        seq_emb = torch.cat((ses, pad), 0)

        return seq_emb

    def get_emd(self, seq):
        batch_seq_emb = []
        for s in seq:
            se = self.get_seq_emb(s)
            batch_seq_emb.append(se[None, :])
        batch_emd = torch.cat(batch_seq_emb, dim=0)

        return batch_emd

    def get_a_emb(self, a):
        a = torch.LongTensor(a)
        a_emb = self.embeddings[a]
        return a_emb

