# Optimus: HuggingFace-Aligned 3D-parallel backend

- flash attention 2 support on training
- flash attention 2 support on left-padding generation with kv cache
- fmha on GQA & MQA
- multi model topology support by mpu context
- more model type for experiment (PPL,RM,...)

TODO:
- GQA & MQA generation (left-padding)
- less model control option
- generator based on non-batch flash attention and self-design cuda fused kernel
- Fixed pipeline model
