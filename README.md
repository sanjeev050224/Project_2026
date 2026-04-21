# Project_2026
Tooth Decay Classification using DenseNet121 with CBAM
**Overview**

Tooth decay (dental caries) is one of the most common oral health problems worldwide. Early detection is important to prevent severe damage, pain, and costly treatments. This project presents an automated Tooth Decay Classification System using deep learning techniques to analyze dental X-ray images and classify the presence or severity of tooth decay.

The proposed model uses DenseNet121 combined with CBAM (Convolutional Block Attention Module) to improve feature extraction and focus on relevant regions in dental radiographs.

**Objectives**
Detect and classify tooth decay from dental X-ray images.
Improve classification performance using attention mechanisms.
Reduce manual diagnostic effort for dentists.
Provide a scalable AI-based solution for oral healthcare support.

**Model Architecture**
Base Model:
DenseNet121
Pretrained on ImageNet
Efficient feature reuse through dense connections
Strong performance with fewer parameters
Attention Module:
CBAM (Convolutional Block Attention Module)
Channel Attention
Spatial Attention
Helps the model focus on important decay regions

Final Pipeline:

Input X-ray Image → Preprocessing → DenseNet121 → CBAM → Fully Connected Layer → Output Class
