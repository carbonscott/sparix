import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embd_size,
                       context_length,
                       head_size,
                       uses_causal_mask  = False,
                       attention_dropout = 0.0,
                       residual_dropout  = 0.0):
        super().__init__()

        self.embd_size         = embd_size
        self.head_size         = head_size
        self.attention_dropout = attention_dropout
        self.residual_dropout  = residual_dropout
        self.uses_causal_mask  = uses_causal_mask

        # Internal variable
        self._num_heads = embd_size // head_size

        # Self-attention layer to update each node by aggregating features from all other nodes...
        # The message-passing based communication happens in another vector space.
        self.proj_q = nn.Linear(self.embd_size, self.embd_size)    # What do I (this node) want?
        self.proj_k = nn.Linear(self.embd_size, self.embd_size)    # What do I have publicly?
        self.proj_v = nn.Linear(self.embd_size, self.embd_size)    # What do I provide to update the entire graph?

        # Store a mask to prevent it from gradient tracking...
        mask = torch.ones(context_length, context_length).triu(diagonal=1).bool()
        self.register_buffer('mask', mask)

        # Linear projection...
        self.proj_linear = nn.Linear(embd_size, embd_size)

        # Use dropout after softmax...
        self.attention_dropout = nn.Dropout(attention_dropout)

        # Use dropout at the end...
        self.residual_dropout = nn.Dropout(residual_dropout)


    def forward(self, x):
        B, T, E = x.shape

        num_heads = self._num_heads
        head_size = self.head_size

        # Linearly project them to a vector space...
        q = self.proj_q(x)   # B, T, E
        k = self.proj_k(x)   # B, T, E
        v = self.proj_v(x)   # B, T, E

        # Changes the view to facilitate scaled dot product within each head...
        q = q.view(B, T, num_heads, head_size).transpose(1, 2)   # (B, num_heads, T, head_size)
        k = k.view(B, T, num_heads, head_size).transpose(1, 2)   # (B, num_heads, T, head_size)
        v = v.view(B, T, num_heads, head_size).transpose(1, 2)   # (B, num_heads, T, head_size)

        # Scaled dot product...
        w = q @ k.transpose(-1, -2)    # (B, num_heads, T, head_size) @ (B, num_heads, head_size, T) ->
                                       # (B, num_heads, T, T)
        w /= torch.sqrt(torch.tensor(head_size))

        # Use causal mask???
        if self.uses_causal_mask:
            # Masking in the decoder to enable causal relation...
            w[:, :,self.mask[:T,:T]] = float('-inf')    # (B, num_heads, :T, :T)   `:T` means upto `T`

        # Obtain the softmax...
        w = w.softmax(dim = -1)    # (B, num_heads, T, T)

        # Aggregate information from all nodes...
        a = w @ v    # (B, num_heads, T, T) @ (B, num_heads, T, head_size) ->
                     # (B, num_heads, T, head_size)

        a = self.attention_dropout(a)

        # Reshape it to (B, T, E)...
        a = a.transpose(2, 1).contiguous()    # (B, num_heads, T, head_size) -> (B, T, num_heads, head_size)
        a = a.view(B, T, E)

        # Linear projection...
        y = self.proj_linear(a)

        # Optional dropout...
        y = self.residual_dropout(y)

        return y




class FeedForward(nn.Module):
    def __init__(self, embd_size, dropout = 0.0):
        super().__init__()

        self.ff_layer = nn.Sequential(
            nn.Linear(    embd_size, 4 * embd_size),
            nn.GELU(),
            nn.Linear(4 * embd_size,     embd_size),
            nn.Dropout(dropout),
        )


    def forward(self, x):
        return self.ff_layer(x)




class TransformerBlock(nn.Module):
    def __init__(self, embd_size, 
                       context_length,
                       num_heads,
                       uses_causal_mask    = False,
                       attention_dropout   = 0.0,
                       residual_dropout    = 0.0,
                       feedforward_dropout = 0.0):
        super().__init__()

        # Define the multi head attention layer to update node position in a sub space using an attention head...
        head_size = embd_size // num_heads
        self.multi_head_att_layer = MultiHeadAttention(embd_size,
                                                       context_length,
                                                       head_size,
                                                       uses_causal_mask  = uses_causal_mask,
                                                       attention_dropout = attention_dropout,
                                                       residual_dropout  = residual_dropout)

        # Define the feedforward layer to add non-linearity to the model...
        self.ff_layer = FeedForward(embd_size, dropout = feedforward_dropout)

        # Define layers to optimize model training...
        self.layer_norm_pre_multi_head  = nn.LayerNorm(embd_size)
        self.layer_norm_pre_feedforward = nn.LayerNorm(embd_size)


    def forward(self, x):
        """
        Arguments:
            x : (B, T, E)

        Returns:
            out : (B, T, E)
        """
        nodes_embd = x

        # ___/ MULTI-HEAD ATTENTION BLOCK \___
        # Go through multi-head attention to update nodes in vector space...
        # ...Pre norm
        nodes_embd_norm = self.layer_norm_pre_multi_head(nodes_embd)

        # ...Attention
        nodes_embd_update = self.multi_head_att_layer(nodes_embd_norm)    # (B, T, E)

        # ...Residual connection (out -> prenorm)
        nodes_embd_update += nodes_embd

        # Learn a better embedding representation by introducing non-linearity...
        # ...Pre norm
        nodes_embd_update_norm = self.layer_norm_pre_feedforward(nodes_embd_update)    # (B, T, E)

        # ...Feed forward
        nodes_embd_better = self.ff_layer(nodes_embd_update_norm)    # (B, T, E)

        # ...Residual connection (out -> prenorm)
        nodes_embd_better += nodes_embd_update

        return nodes_embd_better




class Transformer(nn.Module):
    def __init__(self, tok_size,
                       embd_size,
                       context_length,
                       num_blocks,
                       num_heads,
                       uses_causal_mask    = False,
                       attention_dropout   = 0.0,
                       residual_dropout    = 0.0,
                       feedforward_dropout = 0.0):
        super().__init__()

        # Embed a patch token...
        ## tok_size = Hp * Wp
        self.tok_embd_layer = nn.Linear(tok_size, embd_size)    # (B, T, N) -> (B, T, E)
        ## self.tok_embd_layer = nn.Conv2d(in_channels  = 1,
        ##                                 out_channels = embd_size,
        ##                                 kernel_size  = (Hp, Wp),
        ##                                 stride       = (Hp, Wp))    # (B, T, Hp, Wp) -> (B, T, E)

        # Define positional embedding layer to embed each position to a vector space...
        self.pos_embd_layer = nn.Embedding(context_length, embd_size)

        # Define the multi head attention layer to update node position in a sub space using an attention head...
        head_size = embd_size // num_heads
        self.transformer_block = nn.Sequential(*tuple(
            TransformerBlock(embd_size,
                             context_length,
                             num_heads,
                             uses_causal_mask    = uses_causal_mask,
                             attention_dropout   = attention_dropout,
                             residual_dropout    = residual_dropout,
                             feedforward_dropout = feedforward_dropout) for _ in range(num_blocks)
        ))

        # Define layer norm used in the subsequent prediction head...
        self.layernorm = nn.LayerNorm(embd_size)

        # Prediction head...
        self.pred_head = nn.Linear(embd_size, tok_size)    # (B, T, E) -> (B, T, N)

        # Store a positional tensor...
        self.register_buffer('pos_indices', torch.arange(context_length))


    def forward(self, x):
        """
        N is number of tokens.
        Arguments:
            x : (B, T, Hp * Wp)
        """
        _, T, _ = x.shape

        # ___/ EMBED ALL NODES \___
        nodes_tok_embd = self.tok_embd_layer(x)    # (B, T) -> (B, T, E)
        nodes_pos_embd = self.pos_embd_layer(self.pos_indices[:T])    # (T) -> (T, E)

        nodes_embd = nodes_tok_embd + nodes_pos_embd    # (B, T, E) + (T, E)    =
                                                        # (B, T, E) + (1, T, E) = (B, T, E)

        # ___/ MULTI-HEAD ATTENTION BLOCK \___
        # Go through multi-head attention to update nodes in vector space...
        nodes_embd_better = self.transformer_block(nodes_embd)    # (B, T, E) -> (B, T, E)

        # ___/ PREDICTION HEAD \___
        nodes_embd_better_norm = self.layernorm(nodes_embd_better)
        logits = self.pred_head(nodes_embd_better_norm)    # (B, T, E) -> (B, T, N)

        return logits


    @torch.no_grad()
    def generate_one(self, x):
        """
        Arguments:
            x : (B, T)
        """
        logits = self.forward(x)    # (B, T, N)

        # Look up the logits associated with the last token...
        last_token_logits = logits[:, -1]

        return last_token_logits
