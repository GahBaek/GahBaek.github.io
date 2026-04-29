---
title: "attention is all you need"
layout: single
categories:
  - paper-review
---

<img width="1174" height="1600" alt="image" src="https://github.com/user-attachments/assets/385e9295-7607-4d73-874e-59de43c60619" />


https://github.com/jadore801120/attention-is-all-you-need-pytorch

## Encoder

- Input Embedding
- Positional Encoding
- Multi-Head Attention  
    <img width="1714" height="2096" alt="image" src="https://github.com/user-attachments/assets/5baa4d41-731c-4ca3-b7df-9dc25c017547" />
 
    ```python
    class MultiHeadAttention(nn.Module):
        ''' Multi-Head Attention module '''
    
        def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
            super().__init__()
    
            self.n_head = n_head
            self.d_k = d_k
            self.d_v = d_v
    
            self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
            self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
            self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
            self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
    
            self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
    
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
        def forward(self, q, k, v, mask=None):
    
            d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
            sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
    
            residual = q
    
            # Pass through the pre-attention projection: b x lq x (n*dv)
            # Separate different heads: b x lq x n x dv
            q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
            k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
            v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
    
            # Transpose for attention dot product: b x n x lq x dv
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    
            if mask is not None:
                mask = mask.unsqueeze(1)   # For head axis broadcasting.
    
            q, attn = self.attention(q, k, v, mask=mask)
    
            # Transpose to move the head dimension back: b x lq x n x dv
            # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
            q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
            q = self.dropout(self.fc(q))
            q += residual
    
            q = self.layer_norm(q)
    
            return q, attn
    
    ```
    
    - self attention
        - 같은 문장 내에서 단어들 간의 관계. 즉, 연관성을 고려하여 attention 을 계산하는 방법이다.
    - query, key value
        - q, k, v 각각에 Linear 연산을 거친다.
        - 각각의 차원을 줄여서 병렬 연산에 적합한 구조를 만들려는 목적.
    - attention score
        - query-key 의 내적 (transformers 에서는 대부분 q-k 내적)
            
            ```python
            score = QK^T / sqrt(d_k)
            # softmax 적용해서 attention weight 를 만든다.
            attention_weight = softmax(QK^T / sqrt(d_k))
            # 마지막으로 value 와 곱한다.
            output = softmax(QK^T / sqrt(d_k)) V
            ```
            
        - 행렬곱: 행렬 간의 유사도
- Add & Norm
    - Residual Connection
        - 이전 layer 보다 학습이 덜 되지 않게 이전 학습된 결과에 이번 layer 에서 더 학습할 잔여 학습이 있다면 이를 학습한다.
        - 데이터의 정보가 중간의 layer 을 우회하도록 허용함으로써 network 가 이전 계층의 정보에 접근할 수 있도록 하여, 신경망이 깊어질수록 발생하기 쉬운 “기울기 소실” 등의 리스크를 줄여준다.
        
        ```python
        attention_output = self_attention(x)
        x = x + attention_output
        
        새로운 정보 = attention_output
        기존 정보 = x
        
        둘을 합쳐서 다음 layer로 보냄
        ```
        
    - Layer Normalization
- Feed Forward
    - Linear network + activation function
- Add & Norm

### Decoder

- output Embedding
- Positional Encoding
- Masked Multi-Head Attention
    - 일반 multi-head attention 를 그대로 사용하면 “현재 단어를 예측할 때, 미래 단어에 대한 정보를 사용할 수 있다”는 문제를 야기한다.
    - 따라서 masked multi-head attention 은 모델이 각 시점에서의 정보를 바탕으로 순차적으로 다음 단어를 예측하게 하여, 문장의 일관성과 의미를 유지하며 올바른 문장을 만들어내는 데 중요한 역할을 한다.
    - 종류: causal, padding, span …
- Add & Norm
- Multi-Head Attention
    - Decoder 의 multi-head attention 은 encoder 의 multi-head attention 과 다른 점이 있다.
    - Encoder-Decoder attention 에서는 Query, Key, Value 가 서로 다른 소스에서 생성된다.
    - Query: decoder 의 현재 시점의 출력에서 생성
    - Key, Value: encoder 의 출력에서 얻어지며, Query 는 현재 시점에서 decoder 가 다음에 생성할 단어에 필요한 정보를 찾는 역할을 한다.
    - 이 Encoder-Decoder Attention 덕분에 decoder 는 각 출력이 입력의 어느 부분과 가장 높은 연관성을 가지는지 계산할 수 있다.
- Add & Norm
- Feed Forward
- Add & Norm
- Linear
- Softmax

positional encoding과 attention 만으로는 “동음이의어”와 같은 문제를 해결할 수 없다. → self-attention
