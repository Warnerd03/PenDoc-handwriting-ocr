# PenDoc: Handwritten Character & Word Recognition

This project is a two-stage handwritten text recognition pipeline. The first stage classifies individual handwritten letters using an ensemble of two image classifiers (ResNet18 and DeiT-Tiny). The second stage recognizes full handwritten words directly from the image, using a Convolutional Recurrent Neural Network (CRNN) architecture. CRNN is a Convolutional Neural Network (CNN) feature extractor feeding into a Bidirectional LSTM (BiLSTM), trained with Connectionist Temporal Classification(CTC) loss.

## Pipeline

1. **Kaggle Datasets**:
   - `handwritten-letters` Used to train and test the individual character classification ensemble models 
   - `handwritten-words` Used to train and test the full word image-to-character alignment
2. **Single-Character Image Classifier Models**
   - **ResNet18**: A convolutional network pre-trained on ImageNet, fine-tuned here for single-letter classification. The stem and first residual block stay frozen, as to not unlearn already captured general low-level edge/texture features. The deeper residual blocks and a newly attached classification head are trained on the letter crops.
   - **DeiT-Tiny**: A compact vision transformer pretrained on ImageNet and fine-tuned the same way as ResNet18. This model is an architectural alternative to ResNet18, as it performs self-attention over image patches instead of convolutional filters. The patch embedding and the first half of the transformer blocks are frozen for the same purpose as the ResNet model.
   - Both trained with Adam + OneCycle LR scheduling and early stopping on validation loss.
3. **Ensemble Blend**: 
   - Blend weight on the validation split and applying the best-performing weight to the test split.
4. **Train a CRNN (CNN & BiLSTM) with CTC loss**: 
   - CRNN is a convolutional feature extractor feeds a two-layer bidirectional LSTM that reads the resulting feature map left to right, followed by a linear layer producing per-timestep character probabilities. The model is trained with Connectionist Temporal Classification(CTC) loss, which lets the model learn the alignment between image columns and output characters directly from (image, label) pairs.
5. **Decode CRNN Predictions**:
   - CTC decoding takes the most likely character at each timestep, collapses repeated characters, and drops blank tokens.
6. **Evaluation**:
   - Accuracy, classification report, and confusion matrix for the letter classifiers and ensemble
   - Word accuracy (case-insensitive exact match) and character error rate (CER) for the CRNN.

## Datasets

- [`warrenxnguyen/handwritten-letters`](https://www.kaggle.com/datasets/warrenxnguyen/handwritten-letters)
  (Kaggle): individual handwritten letter images, one subfolder per class (A-Z).
- [`warrenxnguyen/handwritten-words`](https://www.kaggle.com/datasets/warrenxnguyen/handwritten-words)
  (Kaggle): full handwritten word images with corresponding text labels.

## Results

| Model               | Test Accuracy | Notes                                    |
| -------------------- | ------------- | ----------------------------------------- |
| ResNet18              | 95.3%         | Best validation accuracy during training |
| DeiT-Tiny             | 94.59%        | Single-model test accuracy                |
| Ensemble (blend=0.60) | 96.13%        | Blend weight favors ResNet18              |

| Model | Word Accuracy | CER   |
| ----- | -------------- | ----- |
| CRNN  | 85.21%         | 0.040 |

## Findings

- The ensemble outperforms both individual letter classifiers on test accuracy, confirming that ResNet18's convolutional inductive bias and DeiT-Tiny's attention-based features pick up partially complementary signal from the same letter images.
- The tuned blend weight (0.60 toward ResNet18) indicates ResNet18 is the stronger of the two base models on this dataset, with DeiT-Tiny contributing a smaller corrective signal rather than driving the ensemble.
- Per-class errors concentrate on visually similar letter pairs — I/L and G/Q show the lowest precision/recall in both classifiers' reports, consistent with the kind of shape ambiguity a handwriting classifier would be expected to struggle with.
- The CRNN reaches 85.21% exact word accuracy at a character error rate of 4.0%, meaning most of its mistakes are small per-character errors within an otherwise correct word rather than wholesale misreads.
