from codon.base  import *
from codon.motif import MotifA1, MotifV1, VisionEmbedding
from codon.motif.base import CausalLanguageModelOutput, Sampler, KVCache

from typing import Optional, List, Tuple


class MotifA1_VL(BasicModel):
    def __init__(
        self,
        dead_codes: Union[List[int], str] = []
    ):
        super().__init__()
        self.language = MotifA1()
        self.vision = MotifV1()
        
        hidden_dim = getattr(self.language, 'model_dim', None)
        if hidden_dim is None:
            hidden_dim = self.language.token_emb.embedding_dim

        self.vision_emb = VisionEmbedding(
            hidden_dim=hidden_dim,
            dead_codes=dead_codes,
            codebook_dim=15,
            vision_model=self.vision
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[List[torch.Tensor]] = None,
        image_patch_indices: Optional[torch.Tensor] = None,
        mask: torch.Tensor = None,
        start_pos: int = 0,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> CausalLanguageModelOutput:
        '''
        多模态前向传播。
        
        Args:
            input_ids: 文本和占位符序列 [batch_size, seq_len]
            images: 图像 Tensor 列表
            image_patch_indices: [batch_size, num_total_patches] 图像占位符在序列中的绝对物理位置
        '''
        x = self.language.token_emb(input_ids) # [batch_size, seq_len, hidden_dim]

        if images is not None and len(images) > 0 and image_patch_indices is not None:
            batch_size = x.size(0)
            
            if not isinstance(images[0], list) and not isinstance(images[0], torch.Tensor):
                raise ValueError("images formatted incorrectly, expected list of Tensors or nested lists.")
            
            images_by_batch = images if isinstance(images[0], list) else [images] * batch_size

            for b in range(batch_size):
                b_images = images_by_batch[b]
                if not b_images: continue

                b_indices = image_patch_indices[b]
                b_indices = b_indices[b_indices >= 0]
                
                if len(b_indices) == 0: continue

                img_embeddings = []
                for img in b_images:
                    emb = self.vision_emb.embed_image(img.unsqueeze(0)) # [1, patches_per_img, hidden_dim]
                    img_embeddings.append(emb.squeeze(0))

                flat_img_embeddings = torch.cat(img_embeddings, dim=0) # [total_patches, hidden_dim]

                num_to_replace = min(len(b_indices), flat_img_embeddings.size(0))
                if num_to_replace > 0:
                    x[b, b_indices[:num_to_replace]] = flat_img_embeddings[:num_to_replace]

        x = self.language.dropout(x)

        new_kv_cache = [] if use_cache else None
        all_attentions = [] if output_attentions else None
        aux_loss = None

        for i, layer in enumerate(self.language.decoder):
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            out = layer(
                hidden_states=x,
                attention_mask=mask,
                output_attentions=output_attentions,
                position_emb=self.language.position_emb,
                embedding_start=start_pos,
                past_key_value=layer_past,
                use_cache=use_cache
            )
            
            x = out.output
            
            if use_cache:
                new_kv_cache.append(out.past_key_value)
            
            if output_attentions:
                all_attentions.append(out.attention_weights)
            
            if out.aux_loss is not None:
                if aux_loss is None:
                    aux_loss = out.aux_loss
                else:
                    aux_loss += out.aux_loss

        x = self.language.norm(x)
        logits = self.language.proj_out(x)

        return CausalLanguageModelOutput(
            logits=logits,
            past_key_values=new_kv_cache,
            aux_loss=aux_loss,
            attentions=all_attentions
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        images: Optional[List[torch.Tensor]] = None,
        image_patch_indices: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        sampler: Optional[Sampler] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True
    ) -> torch.Tensor:
        '''
        多模态自回归文本生成方法（首 Token Prefill - 续 Token Decode 管道）
        '''
        self.eval()
        if sampler is None:
            sampler = Sampler(temperature=0.7)

        generated = input_ids.clone()
        kv_cache = KVCache() if use_cache else None

        # 1. Prefill
        outputs = self.forward(
            input_ids=input_ids,
            images=images,
            image_patch_indices=image_patch_indices,
            start_pos=0,
            past_key_values=None,
            use_cache=use_cache
        )
        
        logits = outputs.logits[:, -1, :]
        next_token = sampler(logits)
        generated = torch.cat([generated, next_token], dim=-1)

        if use_cache and outputs.past_key_values is not None:
            kv_cache.update(outputs.past_key_values)

        # 2. Decode
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            if use_cache and kv_cache is not None:
                current_pos = kv_cache.current_len
                outputs = self.forward(
                    input_ids=next_token,
                    images=None,
                    image_patch_indices=None,
                    start_pos=current_pos,
                    past_key_values=kv_cache.states,
                    use_cache=True
                )
                kv_cache.update(outputs.past_key_values)
            else:
                outputs = self.forward(
                    input_ids=generated,
                    images=images,
                    image_patch_indices=image_patch_indices,
                    start_pos=0,
                    past_key_values=None,
                    use_cache=False
                )

            logits = outputs.logits[:, -1, :]
            next_token = sampler(logits, input_ids=generated)
            generated = torch.cat([generated, next_token], dim=-1)

        return generated