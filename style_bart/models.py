from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import CrossEntropyLoss

from transformers.models.bart.modeling_bart import (
    BartForConditionalGeneration,
    BartForSequenceClassification, 
    BartForCausalLM,
    BartConfig,
)
from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertForSequenceClassification
)
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.generation_utils import GreedySearchEncoderDecoderOutput

from .custom.generation_utils import GenerationMixin
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput, SequenceClassifierOutput

@dataclass
class SoftGreedySearchEncoderDecoderOutput(GreedySearchEncoderDecoderOutput):
    distributions: torch.FloatTensor = None

class StyleBart(GenerationMixin, BartForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"style_token_ids",
    ]

    def __init__(self, config: BartConfig, style_token_ids):
        super().__init__(config)
        self.register_buffer("style_token_ids", torch.tensor(style_token_ids, dtype=torch.long))

    def generate(
        self,
        styles=None,
        decoder_hard: bool=False,
        return_dict_in_generate: Optional[bool]=None,
        output_scores: Optional[bool]=None,
        **model_kwargs
    ):
        model_kwargs['decoder_start_token_id'] = self.style_token_ids[styles, None]
        model_kwargs['input_ids'] = torch.cat([
            self.style_token_ids[styles, None],
            model_kwargs['input_ids']
        ], dim=1)
        model_kwargs['attention_mask'] = torch.cat([
            torch.ones_like(model_kwargs['attention_mask'][:, :1]),
            model_kwargs['attention_mask']
        ], dim=1)

        if decoder_hard:
            return super().generate(
                return_dict_in_generate=return_dict_in_generate,
                output_scores=output_scores,
                **model_kwargs
            )
        else:
            outputs = super().generate.__wrapped__(
                self,
                output_scores=True,
                return_dict_in_generate=True,
                num_beams=1,
                **model_kwargs
            )

            distributions = torch.stack(outputs.scores, dim=1)
            outputs.scores = outputs.scores if output_scores else None

            if return_dict_in_generate:
                return SoftGreedySearchEncoderDecoderOutput(
                    distributions=distributions,
                    **outputs
                )
            else:
                return distributions

    def forward(
        self,
        input_ids=None,
        decoder_input_ids=None,
        labels=None,
        distributions=None,
        styles=None,
        **kwargs
    ):
        style_token_ids = self.style_token_ids[styles, None]
        if labels is not None:
            decoder_input_ids = torch.cat([style_token_ids, labels[:, :-1]], dim=1)
            labels = labels.masked_fill(labels == self.config.pad_token_id, -100)

        if distributions is not None or input_ids is not None:
            kwargs['attention_mask'] = torch.cat([
                torch.ones_like(kwargs['attention_mask'][:, :1]),
                kwargs['attention_mask']
            ], dim=1)

            if distributions is not None:
                kwargs['inputs_embeds'] = torch.cat([
                    self.get_input_embeddings().weight[style_token_ids],
                    distributions.softmax(-1) @ self.get_input_embeddings().weight
                ], dim=1)
            if input_ids is not None:
                input_ids = torch.cat([
                    style_token_ids,
                    input_ids
                ], dim=1)

        return super().forward(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids, 
            labels=labels, 
            **kwargs
        )

class SoftBertForSentenceClassification(BertForSequenceClassification):
    def __init__(self, config: BertConfig, style_token_ids):
        super().__init__(config)
        self.register_buffer("style_token_ids", torch.tensor(style_token_ids, dtype=torch.long))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        styles=None,
        distributions=None,
    ):
        style_token_ids = self.style_token_ids[styles, None]
        attention_mask = torch.cat([
            torch.ones_like(attention_mask[:, :1]),
            attention_mask
        ], dim=1)

        if input_ids is not None:
            input_ids = torch.cat([style_token_ids, input_ids], dim=1)
        if distributions is not None:
            inputs_embeds = torch.cat([
                self.get_input_embeddings()(style_token_ids),
                distributions.softmax(-1) @ self.get_input_embeddings().weight,
            ], dim=1)
            input_ids = None

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(label_smoothing=.1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class SoftBartForSentenceClassification(BartForSequenceClassification):
    _keys_to_ignore_on_load_missing = [
        r"style_token_ids",
        r"classification_head\.dense\.bias",
        r"classification_head\.dense\.weight",
        r"classification_head\.out_proj\.bias",
        r"classification_head\.out_proj\.weight",
    ]
    def __init__(self, config: BartConfig, style_token_ids):
        super().__init__(config)
        self.register_buffer("style_token_ids", torch.tensor(style_token_ids, dtype=torch.long))

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        distributions=None,
        attention_mask=None,
        labels=None,
        styles=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        style_token_ids = self.style_token_ids[styles, None]
        attention_mask = torch.cat([
            torch.ones_like(attention_mask[:, :1]),
            attention_mask
        ], dim=1)

        if input_ids is not None:
            input_ids = torch.cat([style_token_ids, input_ids], dim=1)
            decoder_input_ids = input_ids[:, :-1]
        if distributions is not None:
            inputs_embeds = torch.cat([
                self.get_input_embeddings()(style_token_ids),
                distributions.softmax(-1) @ self.get_input_embeddings().weight,
            ], dim=1)
            decoder_inputs_embeds = inputs_embeds[:, :-1, :]

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        eos_mask = input_ids[:, 1:].eq(self.config.eos_token_id)
        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )
        elif inputs_embeds is not None:
            input_ids = None
            decoder_input_ids = None
        else:
            decoder_inputs_embeds = None

        outputs = self.model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # last hidden state


        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(label_smoothing=.1)
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

class GPT2LMHeadModel(GPT2LMHeadModel):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        input_ids = torch.cat([
            torch.scalar_tensor(
                2, dtype=torch.long, device=input_ids.device
            )[None, None].expand((input_ids.shape[0], -1)),
            input_ids
        ], dim=1)
        attention_mask = torch.cat([
            torch.ones_like(attention_mask[:, :1]),
            attention_mask
        ], dim=1)
        if labels is not None:
            labels = torch.cat([
                torch.scalar_tensor(
                    2, dtype=torch.long, device=input_ids.device    # BartConfig().decoder_start_token_id
                )[None, None].expand((input_ids.shape[0], -1)), 
                labels
            ], dim=1)
            labels = labels.masked_fill(labels == 1, -100)          # BartConfig().pad_token_id
        
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        outputs.logits = outputs.logits[:, :-1, :]
        return outputs

class BartForCausalLM(BartForCausalLM):
    def __init__(self, config: BartConfig, encoder_hidden_states: Optional[torch.Tensor]=None):
        super().__init__(config)

        if encoder_hidden_states is None:
            encoder_hidden_states = torch.empty((1, 3, config.d_model))
            torch.nn.init.normal_(encoder_hidden_states)

        self.register_buffer("encoder_hidden_states", encoder_hidden_states)

    def forward(
        self,
        input_ids,
        labels=None,
        **kwargs
    ):
        input_ids = torch.cat([
            torch.scalar_tensor(
                self.config.decoder_start_token_id, dtype=torch.long, device=input_ids.device
            )[None, None].expand((input_ids.shape[0], -1)),
            input_ids[:, :-1]
        ], dim=1)

        if labels is not None:
            labels = labels.masked_fill(labels == self.config.pad_token_id, -100)

        return super().forward(
            input_ids=input_ids,
            labels=labels, 
            encoder_hidden_states=
                self.encoder_hidden_states.expand(input_ids.shape[0], -1, -1) \
                    if self.encoder_hidden_states is not None else None,
            **kwargs
        )
