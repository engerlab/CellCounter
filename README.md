# Automated Colony Counting for Clonogenic Cell Survival Assays

Data can be made available upon request.

## Abstract

**Background and Objective:** Assessment of cancer cell radiosensitivity is essential for evaluating the effectiveness of radiotherapy. The clonogenic assay remains the gold standard for quantifying radiosensitivity by enumerating cells with replicative potential\textit{in vitro} following radiation exposure. However, colony counting is labor-intensive and subject to inter-observer variability. This study aimed to develop and evaluate an automated colony counting method for clonogenic assay images.

**Methods:** We developed a classical computer vision-based pipeline for automated colony counting and evaluated it on 787 clonogenic assay images of the HCT116 colorectal cancer cell line. The method combines targeted preprocessing with an augmented marker-based watershed framework to segment and enumerate colonies. The pipeline was designed for settings with limited annotated data, variable imaging conditions, and the absence of microscopic-resolution images of individual cells, and produces interpretable segmentation masks that can serve as pseudo-annotations for downstream deep learning-based methods.

**Results:** Quantitative evaluation across four assay morphology categories achieved an average precision of 0.961, recall of 0.779, and an overall F1 score of 0.856. Performance remained high for assays with sparse or moderately separated colonies (F1 scores up to 0.891) but decreased in densely clustered assays due to increased colony overlap. The proposed pipeline achieved higher overall precision and recall benchmarked against the Segment Anything Model, and produced fewer false detections arising from background artifacts.

**Conclusions:** These results demonstrate that classical computer vision methods can provide a robust and interpretable solution for automated clonogenic assay analysis under data- and annotation-limited conditions. 

## Keywords
Computer vision, segmentation,  colony counting, clonogenic assay

## Authors

Mona Wang (wanrong.wang@mail.mcgill.ca) <br />
Joanna Li <br />
PI: Dr. Shirin A. Enger <br />
Supervisor: Dr. Laya Rafiee Sevyeri
