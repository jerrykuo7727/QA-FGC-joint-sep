import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertModel, BertPreTrainedModel


class BertQAJoint(BertPreTrainedModel):
    def __init__(self, config):
        super(BertQAJoint, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.se_outputs = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):
        ''' Single-span loss. '''
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
    
    
    def predict_biloss(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                       start_positions=None, end_positions=None, SE_positions=None):
        ''' Single-span and SE loss. '''
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = outputs[0]

        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)        

        logits = self.se_outputs(sequence_output)
        logits = logits.squeeze(-1)

        outputs = (start_logits, end_logits, logits, ) + outputs[2:]
        if start_positions is not None and end_positions is not None and SE_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(SE_positions.size()) > 1:
                SE_positions = SE_positions.squeeze(-1)
            
            # Single-span loss
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            # Sentence-of-span loss
            SE_ignored_index = logits.size(1)
            SE_positions.clamp_(0, SE_ignored_index)
            SE_loss_fct = CrossEntropyLoss(ignore_index=SE_ignored_index)
            SE_loss = SE_loss_fct(logits, SE_positions)

            total_loss = start_loss + end_loss + 2 * SE_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


    def predict_triloss(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                        start_positions=None, end_positions=None, SE_positions=None, sep_mask=None, shint_mask=None):
        ''' Single-span, SE and shint loss. '''
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        sequence_output = outputs[0]

        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        logits = self.se_outputs(sequence_output)
        logits = logits.squeeze(-1)

        outputs = (start_logits, end_logits, logits, ) + outputs[2:]
        if start_positions is not None and end_positions is not None and \
           SE_positions is not None and \
           sep_mask is not None and shint_mask is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(SE_positions.size()) > 1:
                SE_positions = SE_positions.squeeze(-1)

            # Single-span loss
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            # Sentence-of-span loss
            SE_ignored_index = logits.size(1)
            SE_positions.clamp_(0, SE_ignored_index)
            SE_loss_fct = CrossEntropyLoss(ignore_index=SE_ignored_index)
            SE_loss = SE_loss_fct(logits, SE_positions)

            # Supporting-evidence loss
            masked_logits = torch.masked_select(logits, sep_mask)
            masked_targets = torch.masked_select(shint_mask, sep_mask)
            shint_loss_fct = BCEWithLogitsLoss()
            shint_loss = shint_loss_fct(masked_logits, masked_targets)

            total_loss = start_loss + end_loss + SE_loss + shint_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
