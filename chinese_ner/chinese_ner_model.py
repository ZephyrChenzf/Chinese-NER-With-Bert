"""
@autor: chenzf
@file: chinese_ner_model.py
@time: 2019/4/9 1:08 PM

"""
import torch
from torch import nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel,BertModel

class BertChineseNER(BertPreTrainedModel):
    def __init__(self,config,num_labels):
        super(BertChineseNER,self).__init__(config)
        self.num_labels=num_labels
        self.bert=BertModel(config)
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        self.classifier=nn.Linear(config.hidden_size,num_labels)
        self.apply(self.init_bert_weights)


    def forward(self,input_ids,token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output=self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


