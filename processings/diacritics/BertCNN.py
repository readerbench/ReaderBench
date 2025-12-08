# BertCNN training implementation has been migrated to PyTorch.
# The original TensorFlow/Keras implementation has been removed.
# Training code is not used at inference time - DiacriticsRestoration.py handles inference.
#
# To retrain the model in PyTorch, implement a PyTorch equivalent of the BertCNN architecture
# described in the original file and train from scratch or convert weights from the saved Keras model.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import rb.processings.diacritics.utils as utils
import functools


class BertCNN(nn.Module):
    """
    PyTorch implementation of Bert + Character Level CNN for diacritics classification.
    Architecture mirrors the original Keras model.
    """

    def __init__(self, window_size, alphabet_size, embedding_size, conv_layers, fc_hidden_size,
                 num_of_classes, batch_max_sentences, batch_max_windows,
                 bert_trainable, cnn_dropout_rate, bert_model_name, learning_rate):
        super().__init__()
        self.window_size = window_size
        self.alphabet_size = alphabet_size
        self.embedding_size = embedding_size
        self.conv_layers_config = conv_layers
        self.total_number_of_filters = functools.reduce(lambda x, y: x + y[0], conv_layers, 0)
        self.num_of_classes = num_of_classes
        self.cnn_dropout_rate = cnn_dropout_rate
        self.fc_hidden_size = fc_hidden_size
        self.batch_max_sentences = batch_max_sentences
        self.batch_max_windows = batch_max_windows

        # BERT encoder
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        if not bert_trainable:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Character mask embedding (non-trainable)
        mask_weights = self._build_embedding_mask()
        self.char_mask_emb = nn.Embedding(alphabet_size, num_of_classes)
        self.char_mask_emb.weight = nn.Parameter(
            torch.tensor(mask_weights, dtype=torch.float), requires_grad=False
        )

        # Character sequence embedding
        self.char_emb = nn.Embedding(alphabet_size, embedding_size)

        # Convolution layers
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_size, num_filters, kernel_size=filter_width)
            for num_filters, filter_width in conv_layers
        ])

        self.dropout = nn.Dropout(cnn_dropout_rate)

        cnn_out_size = self.total_number_of_filters + embedding_size  # + middle char embedding

        # Fusion + classifier
        self.hidden = nn.Linear(self.bert_hidden_size + cnn_out_size, fc_hidden_size)
        self.output = nn.Linear(fc_hidden_size, num_of_classes)

    def _build_embedding_mask(self):
        weights = np.ones((self.alphabet_size, self.num_of_classes))
        weights[2] = [1, 1, 1, 0, 0]   # a -> a, ă, â
        weights[10] = [1, 0, 0, 1, 0]  # s -> s, ș
        weights[13] = [1, 0, 0, 0, 1]  # t -> t, ț
        weights[16] = [1, 0, 1, 0, 0]  # i -> i, î
        return weights

    def forward(self, bert_input_ids, bert_segment_ids, token_ids, sent_ids, mask, char_windows):
        """
        Args:
            bert_input_ids:   (batch, max_sent, bert_max_seq_len)
            bert_segment_ids: (batch, max_sent, bert_max_seq_len)
            token_ids:        (batch, max_windows)
            sent_ids:         (batch, max_windows)
            mask:             (batch, max_windows)
            char_windows:     (batch, max_windows, window_size)
        """
        batch_size = bert_input_ids.shape[0]
        max_sent = bert_input_ids.shape[1]
        seq_len = bert_input_ids.shape[2]

        # BERT forward pass over all sentences in batch
        flat_ids = bert_input_ids.view(-1, seq_len)
        flat_seg = bert_segment_ids.view(-1, seq_len)
        attention_mask = (flat_ids != 0).long()
        bert_out = self.bert(input_ids=flat_ids, attention_mask=attention_mask, token_type_ids=flat_seg)
        # (batch*max_sent, seq_len, hidden_size)
        bert_out = bert_out.last_hidden_state.view(batch_size, max_sent, seq_len, self.bert_hidden_size)

        # Gather BERT token embeddings per window
        b_idx = torch.arange(batch_size, device=bert_input_ids.device).unsqueeze(1).expand(-1, self.batch_max_windows)
        bert_tokens = bert_out[b_idx, sent_ids, token_ids]  # (batch, max_windows, hidden_size)

        # CNN character embeddings
        flat_windows = char_windows.view(-1, self.window_size)  # (batch*max_windows, window_size)
        x = self.char_emb(flat_windows)  # (batch*max_windows, window_size, emb_size)
        middle_char_emb = x[:, (self.window_size - 1) // 2, :]  # (batch*max_windows, emb_size)

        x = x.transpose(1, 2)  # (batch*max_windows, emb_size, window_size) for Conv1d
        conv_outputs = []
        for conv in self.convs:
            c = torch.tanh(conv(x))  # (batch*max_windows, filters, new_len)
            c = c.max(dim=2).values  # global max pooling
            conv_outputs.append(c)

        cnn_out = torch.cat(conv_outputs, dim=1)  # (batch*max_windows, total_filters)
        cnn_out = self.dropout(cnn_out)
        cnn_out = torch.cat([cnn_out, middle_char_emb], dim=1)  # + middle char embedding
        cnn_out = self.dropout(cnn_out)
        cnn_out = cnn_out.view(batch_size, self.batch_max_windows, -1)

        # Fuse BERT and CNN
        fused = torch.cat([bert_tokens, cnn_out], dim=-1)  # (batch, max_windows, bert+cnn)
        hidden = F.relu(self.hidden(fused))
        logits = self.output(hidden)  # (batch, max_windows, num_classes)
        predictions = torch.softmax(logits, dim=-1)

        # Apply character mask
        flat_char_centers = char_windows[:, :, (self.window_size - 1) // 2]  # (batch, max_windows)
        char_mask = self.char_mask_emb(flat_char_centers)  # (batch, max_windows, num_classes)
        masked_predictions = predictions * char_mask

        # Flatten to (batch*max_windows, num_classes) and return alongside mask
        flat_preds = masked_predictions.view(-1, self.num_of_classes)
        flat_mask = mask.view(-1, 1)
        return flat_preds, flat_mask


def weighted_categorical_crossentropy(weights, num_of_classes):
    """Returns a loss function compatible with the PyTorch BertCNN."""
    weight_tensor = torch.tensor(weights, dtype=torch.float)

    def loss(y_pred, y_true):
        # y_pred: (N, C), y_true: (N, C) one-hot
        y_pred = y_pred / (y_pred.sum(dim=-1, keepdim=True) + 1e-8)
        y_pred = y_pred.clamp(1e-7, 1.0)
        w = weight_tensor.to(y_pred.device)
        return -(y_true * torch.log(y_pred) * w).sum(dim=-1).mean()

    return loss
