# Readme

## 2022-03-29

1. Add `sigmoid()` on reference box before positional embedding.

2. Normalize theta $\in [\frac{-\pi}{2},\frac{\pi}{2})$ to [0, 1).
3. Change the smooth L1 loss of polys in format of eight params  to L1 loss of oboxes in format of five params.
4. grad detach on reference box of every decoder layer except the first layer



## 2022-04-06
1. disable the gaussian co-attention
2. back to sam-detr without roi aign
