import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
import dataloader
import modules

from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification
# from pytorch_pretrained_bert.modeling import BertForSequenceClassification


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

class DatasetBERT(Dataset):
    """
    Dataset class for Natural Language Inference datasets.
    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, text_a) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        return features

    def transform_text(self, data, batch_size=32):
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(data,
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        return eval_dataloader

class ClassificationBERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32):
        super(ClassificationBERT, self).__init__()
        print(pretrained_dir)
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()

        # construct dataset loader
        self.dataset = DatasetBERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data, batch_size=32):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []
        # print (text_data)
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)

class ClassificationModel(nn.Module):
    def __init__(self, embedding, hidden_size=150, depth=1, dropout=0.3, cnn=False, nclasses=2):
        super(ClassificationModel, self).__init__()
        self.cnn = cnn
        self.drop = nn.Dropout(dropout)
        self.emb_layer = modules.EmbeddingLayer(
            embs = dataloader.load_embedding(embedding)
        )
        self.word2id = self.emb_layer.word2id

        if cnn:
            self.encoder = modules.CNN_Text(
                self.emb_layer.n_d,
                widths = [3,4,5],
                filters=hidden_size
            )
            d_out = 3*hidden_size
        else:
            self.encoder = nn.LSTM(
                self.emb_layer.n_d,
                hidden_size//2,
                depth,
                dropout = dropout,
                # batch_first=True,
                bidirectional=True
            )
            d_out = hidden_size
        # else:
        #     self.encoder = SRU(
        #         emb_layer.n_d,
        #         args.d,
        #         args.depth,
        #         dropout = args.dropout,
        #     )
        #     d_out = args.d
        self.out = nn.Linear(d_out, nclasses)

    def forward(self, input):
        if self.cnn:
            input = input.t()
        emb = self.emb_layer(input)
        emb = self.drop(emb)

        if self.cnn:
            output = self.encoder(emb)
        else:
            output, hidden = self.encoder(emb)
            # output = output[-1]
            output = torch.max(output, dim=0)[0].squeeze()

        output = self.drop(output)
        return self.out(output)

    def text_pred(self, text, batch_size=32):
        batches_x = dataloader.create_batches_x(
            text,
            batch_size, ##TODO
            self.word2id
        )
        outs = []
        with torch.no_grad():
            for x in batches_x:
                x = Variable(x)
                if self.cnn:
                    x = x.t()
                emb = self.emb_layer(x)

                if self.cnn:
                    output = self.encoder(emb)
                else:
                    output, hidden = self.encoder(emb)
                    # output = output[-1]
                    output = torch.max(output, dim=0)[0]

                outs.append(F.softmax(self.out(output), dim=-1))

        return torch.cat(outs, dim=0)