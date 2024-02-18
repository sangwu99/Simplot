import torch 
import torch.nn as nn 

from transformers import Pix2StructForConditionalGeneration, Pix2StructVisionModel

class Simplot(nn.Module):
    def __init__(self, args, margin=0, lambda_=0.1):
        super(Simplot, self).__init__()
        
        self.args = args
        self.base = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
        self.teacher_encoder = Pix2StructVisionModel(config = self.base.config.vision_config)
        self.table_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
        self.triplet_loss = nn.TripletMarginLoss(margin)
        self.lambda_ = lambda_
        self.phase = args.phase
        
        if self.phase == 2:
            self.teacher_encoder.load_state_dict(self.base.encoder.state_dict())
            
            for params in self.teacher_encoder.parameters():
                params.requires_grad = False
            
            for params in self.base.decoder.parameters():
                params.requires_grad = False
        
    def forward(self, flattened_patches, attention_mask, labels = None):
        outputs = self.base(flattened_patches = flattened_patches,
                            attention_mask = attention_mask,
                            labels = labels)
        
        return outputs.loss
    
    def forward_phase_2(self, flattened_patches, attention_mask, labels, batch_size):
        anchor = self.base.encoder(flattened_patches = flattened_patches[:batch_size],
                                   attention_mask = attention_mask[:batch_size])[0]
        hidden_states = self.teacher_encoder(flattened_patches = flattened_patches[batch_size:],
                                       attention_mask = attention_mask[batch_size:])[0]
        
        hidden_states = torch.cat([anchor, hidden_states], dim = 0)
        repeated_labels = labels.repeat(3, 1)
        decoder_input_ids = self.base._shift_right(repeated_labels)
        decoder_attention_mask = decoder_input_ids.ne(self.base.config.pad_token_id).float()
        decoder_attention_mask[:, 0] = 1
        decoder_outputs = self.base.decoder(input_ids = decoder_input_ids,
                                            attention_mask=decoder_attention_mask,
                                            encoder_hidden_states=hidden_states,
                                            encoder_attention_mask=attention_mask,
                                            labels = repeated_labels, 
                                            output_hidden_states = True)
        
        triplet_loss = self.triplet_loss(decoder_outputs.hidden_states[-1][:batch_size],
                                         decoder_outputs.hidden_states[-1][batch_size:2*batch_size].detach(),
                                         decoder_outputs.hidden_states[-1][2*batch_size:].detach())
        
        logits = decoder_outputs.logits[:batch_size]
        table_loss = self.table_loss(logits.contiguous().view(-1, logits.size(-1)), labels.contiguous().view(-1))
        loss = ((self.lambda_ * triplet_loss) + ((1 - self.lambda_) * table_loss))
        
        return loss
    
    def generate(self, flattened_patches, attention_mask):
        return self.base.generate(flattened_patches = flattened_patches, attention_mask = attention_mask, max_length = 800)