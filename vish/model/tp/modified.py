from torch import nn
from transformers.models.vit import ViTModel

from vish.model.tp.tp_vit import TPVitImageClassification


class TPDualModifiedVit(nn.Module):
    def __init__(
        self,
        fine_model: TPVitImageClassification,
        broad_model: ViTModel,
        debug=True,
    ):
        super().__init__()
        self.emb_dim = fine_model.emb_dim
        self.embeddings = broad_model.embeddings
        self.broad_encoders = broad_model.encoder.layer
        self.fine_encoders = fine_model.transformer_encoder.tf_blocks
        self.num_classification_heads = fine_model.num_classification_heads
        self.mlp_heads = fine_model.mlp_heads
        # self.ln_fine = nn.LayerNorm(self.emb_dim)
        # self.ln_broad = nn.LayerNorm(self.emb_dim)
        self.debug = debug

        self.log("Model INIT complete")

    def to_logits(self, additional_embeddings):
        logits = [
            self.mlp_heads[idx](additional_embeddings[:, idx])
            for idx in range(self.num_classification_heads)
        ]
        return logits

    @staticmethod
    def log(f_string):
        if False:
            print(f_string)

    def get_encoded_inputs(self, pixel_values, interpolate_pos_encoding):
        encodings = self.embeddings(pixel_values, None, interpolate_pos_encoding)
        self.log(f"Image as Encoding from pretrained model: {encodings.shape}")
        return encodings

    def get_fine_encoder_layers(self):
        x = self.fine_encoders
        self.log(f"Number of TP Encoder {len(x)}")
        return x

    def get_broad_encoder_layers(self):
        x = self.broad_encoders
        self.log(f"Number of ViT encoders blocks {len(x)}")
        return x

    def forward(self, pixel_values):
        broad_t, fine_t = self.get_model_input_encodings(False, pixel_values)
        broad_t, fine_t = self.pass_via_encoders(broad_t, fine_t)
        broad_embedding, fine_embedding, fine_logits = self.get_model_outputs(
            broad_t, fine_t
        )
        return broad_embedding, fine_embedding, fine_logits

    def get_model_outputs(self, broad_t, fine_t):
        # Embeddings
        broad_embedding = broad_t[:, :1, :]
        fine_embedding = fine_t[:, :1, :]
        fine_logits = self.get_fine_logits(fine_embedding)
        self.log(f"Broad Embedding Shape: {broad_embedding.shape}")
        self.log(f"Fine Embedding Shape: {fine_embedding.shape}")
        self.log(f"Fine Logits Shape: {fine_logits.shape}")
        return broad_embedding, fine_embedding, fine_logits

    def get_fine_logits(self, fine_embedding):
        self.log(f"Fine Embedding Shape: {fine_embedding.shape}")
        fine_logits = self.to_logits(fine_embedding)
        if isinstance(fine_logits, (list, tuple)):
            fine_logits = fine_logits[-1]
        return fine_logits

    def pass_via_encoders(self, broad_t, fine_t):
        f_encoders = self.get_fine_encoder_layers()
        b_encoders = self.get_broad_encoder_layers()
        for idx, (b_block, f_block) in enumerate(zip(b_encoders, f_encoders)):
            self.log(f"Processing Block {idx} in Encoders")
            broad_t = b_block(broad_t)[0]
            fine_t = f_block(fine_t, x_ext=broad_t, mask=None)
        return broad_t, fine_t

    def get_model_input_encodings(self, interpolate_pos_encoding, pixel_values):
        fine_t = broad_t = self.get_encoded_inputs(
            pixel_values, interpolate_pos_encoding
        )
        self.log(f"Image Encoding Shape: {broad_t.shape}")
        return broad_t, fine_t
