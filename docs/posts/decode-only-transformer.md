---
categories:
  - mlsys
  - note
date:
  created: 2025-11-15
---



# Some Notes about Transformer in LLM

最近想入门一下大模型的serving和inference。可能Transformer的架构是一个好的切入点。
<!-- more -->
第一步，transformer需要将输入的prompt变成input embeddings

## 1. Tokenizer

word embedding uses various methods (tiktoken in Llama), while the position emdedding is different.

in original Transformer, 

$$
PE{(t, 2i)} = \sin\left(\frac{t}{10000^{2i/d{\text{model}}}}\right)
$$

$$
PE{(t, 2i+1)} = \cos\left(\frac{t}{10000^{2i/d{\text{model}}}}\right)
$$

, where $i$ is the dimension position in a token, $t$ is the token's absolute position in the query.

transformer takes the sum of $\mathbf{p}_t$ (Word Embedding) and  $\mathbf{e}_t$ (Word Embedding) as the input $\mathbf{z}_t$: 

$$
\mathbf{z}t = \mathbf{e}t + \mathbf{p}_t
$$


**Llama** uses another kind of Position Embedding, **RoPE** to describe the postion, and it works on the $Q \& K$ caculation by **rotation** not the input generation by addition.

For a token at the absolute position $m$, the rotation matrix $R_m$, it will rotate every pairs $(q_1, q_2)$ in $Q_m$. the number of the pairs is $d_{model}/2$, 

$$
Q'_m = R_mQ_m
$$


where every pair $(q_1, q_2)$ is rotated by $\theta_i$, 

$$
\begin{pmatrix} q'_1 \\ q'_2 \end{pmatrix} = 
\begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} q_1 \\ q_2 \end{pmatrix}
$$


Where the $\theta$ is calculated by $i$ and $d_{model}$, $\theta_i = \frac{1}{b^{2i / d}}, b = 10000$


Notice that there is no additional computation compared with original implementation (easy to prove).

Now look into the implementation in Llama: 

```python
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

```

第二步是Query、Key、Value的计算

## 2. Q, K, V Calculation





