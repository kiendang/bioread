from transformers import (
    BertForPreTraining, BertForTokenClassification, BertModel
)

import torch
from torch import nn
from torch.nn import BCELoss


class BertForClozeQA(BertForPreTraining):
    def __init__(self, config):
        super(BertForClozeQA, self).__init__(config)

        self.bert = BertModel(config)
        self.cloze_qa_outputs = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                p_mask=None, candidates=None, start_positions=None):
        batch_dim, segment_dim, token_dim = input_ids.size()
        input_ids_flat, attention_mask_flat, token_type_ids_flat = tuple(
            torch.flatten(tensor, 0, 1)
            for tensor in (input_ids, attention_mask, token_type_ids)
        )

        outputs = self.bert(input_ids_flat,
                            attention_mask=attention_mask_flat,
                            token_type_ids=token_type_ids_flat)

        sequence_output, *_ = outputs

        start_logits = self.cloze_qa_outputs(sequence_output).squeeze(-1)
        start_logits_concat = start_logits.reshape(batch_dim, -1)

        p = torch.softmax(start_logits_concat, dim=1)
        pm = p * p_mask
        pm = pm / torch.sum(pm, dim=1).expand_as(pm)
        pc = torch.bmm(pm.unsqueeze(-1), candidates)

        outputs = (start_logits, ) + outputs[2:]

        if start_positions is not None:
            loss_fct = BCELoss()
            loss = loss_fct(pc, start_positions)

            outputs = (loss, ) + outputs

        return outputs
