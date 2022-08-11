# GMP-VIS

This repo will release an official PaddlePaddle implementation for paper: A Graph Matching Perspective with Transformers on Video Instance Segmentation.

# Abstract
In this work, we study the challenge of video Instance Segmentation (VIS), which needs to track and segment multiple objects in videos automatically. We introduce a novel network from a graph matching perspective to formulate VIS, called GMP-VIS. Unlike traditional tracking-by-detection paradigm or bottom-up generative solutions, GMP-VIS uses a novel, learnable graph matching Transformer to predict the instances by heuristically learning the spatial-temporal relationships. Specifically, we take advantage of the powerful Transformer and exploit temporal feature aggregation to capture long-term temporal information across frames implicitly. After generating instance proposals for each frame, the difﬁcult instance association problem is cast as a more leisurely, differentiable graph matching task. The graph matching mechanism performs the data association between current and historical frames based on the proposed instance feature, which can better infer the deformations and obscured foreground instances. Building graph-level annotation during network training allows our GMP-VIS to mine more structural supervision signiﬁcantly distinguished from current VIS solutions. Our extensive experiments over three representative benchmarks, including YouTube-VIS19, YouTube-VIS21, and OVIS, demonstrate that GMP-VIS outperforms the current alternatives by a large margin.
![the overall framework](./arch-.pdf)
