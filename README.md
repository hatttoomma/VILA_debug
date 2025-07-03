Train/test time analyse  
Values correspond the the cosine similarity on each position in dimension 0

## 1.inputs media before ViT as a fixed random tensor
#### ViT
ViT input (image_{}):

    min: 0.999993

ViT output(image_features_{}):

    Min:  0.888047

    Mean: 0.999577

    \>0.95:     17404 (100.0%)

    0.8-0.95:          4 (0.0%)

#### mm_projector
mlp output(features_after_mm_projector_{}):

    Min:  0.966186

    Mean: 0.999402

#### llm:(ignore last 50 tokens to avoid differencies caused by missing of \n after \<image\> token in evaluation)
llm output(llm_outputs_{}):

    Mean cosine similarity: 0.980196

    Min cosine similarity: -0.248717

    Different tokens (< 0.95): 264

    Total tokens: 3079

## 2. use the first sample in vstar as the input

#### ViT
ViT input:

    Min:  0.999998

ViT output:

    \>0.95):     17072 (98.1%)

    (0.8-0.95]:        294 (1.7%)

    (0.5-0.8]:         42 (0.2%)

#### mm_projector
mlp output:

    \>0.95):     3046 (99.2%)

    (0.8-0.95]:         20 (0.7%)

    (0.5-0.8]:          6 (0.2%)

#### llm:(ignore last 50 tokens to avoid differencies caused by missing of \n after \<image\> token in evaluation)
llm output:

    Mean cosine similarity: 0.970199

    Min cosine similarity: -0.295569

    Max cosine similarity: 0.999983

    Different tokens (< 0.95): 365

    Total tokens: 3079


## 3. fixed inputs for vit,mlp,llm
vit:

    Min:  0.955817

mlp:

    Min:  0.999991

llm:

    Min: 0.987829