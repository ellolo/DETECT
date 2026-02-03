import torch
from typing import Dict
from .base import Encoder
from .bert import BERTEncoder

from transformers import RobertaModel, AutoTokenizer, AutoConfig, AutoModel

#from lens.lens.encoders.base import BaseEncoder


class RoBERTaEncoder(BERTEncoder):
    """Encodes sentences using a RoBERTa model."""

    def __init__(self, pretrained_model: str, load_pretrained_weights: bool = True) -> None:
        super(Encoder, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_model)
        config.output_hidden_states = True # Ensure this is set in the config
        self.model = AutoModel.from_pretrained(
            pretrained_model, add_pooling_layer=False
        )
        self.model.encoder.output_hidden_states = True
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        

    """
    def __init__(self, model_name: str):
        super().__init__()
        # Load configuration and explicitly ensure output_hidden_states is True
        # This is the most robust way to ensure the model returns all expected outputs.
        config = AutoConfig.from_pretrained(model_name)
        config.output_hidden_states = True # Ensure this is set in the config
        self.model = RobertaModel.from_pretrained(model_name, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._device = None
    """
    
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # Explicitly request output_hidden_states=True and return_dict=True for robustness
        #model_output = self.model(
        #    input_ids=input_ids,
        #    attention_mask=attention_mask,
        #    output_hidden_states=True, # Explicitly request hidden states
        #    return_dict=True # Always return dictionary for explicit access
        #)
        
        last_hidden_states, _, all_layers = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True, # Explicitly request hidden states
            return_dict=True # Always return dictionary for explicit access
        )

        # Access outputs by name
        # last_hidden_state = model_output.last_hidden_state # Not used directly in this snippet
        # pooler_output = model_output.pooler_output # Not used directly in this snippet
        #all_layers = model_output.hidden_states # This is a tuple of all layer outputs

        # Use the mean of the last layer's hidden states as the sentence embedding.
        #sentence_embedding = all_layers[-1].mean(dim=1)

        #return {"sentence_embedding": sentence_embedding}
        return {
            "sentemb": last_hidden_states[:, 0, :],
            "wordemb": last_hidden_states,
            "all_layers": all_layers,
            "attention_mask": attention_mask,
        }
