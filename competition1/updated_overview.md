> **‼️ 📥 Access the [real IoT time-series data](https://github.com/cruiseresearchgroup/DIEF_BTS?tab=readme-ov-file) released as a part of the [NeurIPS 2024 pape](https://neurips.cc/virtual/2024/poster/97839)r.**
> 
> **📚 Make your first submission with ease [using this starter-kit](https://github.com/cruiseresearchgroup/DIEF_BTS/tree/main/competition1).**

# 💡Building Smarter Buildings

A global challenge to automate building data classification, unlocking more intelligent, energy-efficient buildings for a sustainable future.

Buildings are among the largest energy consumers in the modern world, making energy efficiency essential. However, managing building system data across diverse facilities is time-consuming and costly due to inconsistent data formats. This **time-series classification problem** invites you to transform building management by creating a solution that automatically classifies building data, promoting standardised, energy-efficient management for a more sustainable world.

# 🏢 Introduction

Buildings significantly influence our comfort, health, and environment, accounting for a substantial portion of global energy use and emissions. Effective building management through technology is essential in the fight against climate change. The Brick schema, a standardised metadata schema, provides a potential solution but is costly and labour-intensive to implement manually. Brick by Brick aims to change this by automating the data classification process making technological solutions more accessible and sustainable.

The Brick by Brick challenge seeks to overcome these barriers by automating the data classification. Participants will tackle the critical task of classifying time-series data from IoT devices following the [Brick schema](https://brickschema.org/), advancing the scalability and efficiency of smart buildings.

# 🧩 Problem Statement

Participants must classify building data streams into categories defined by the Brick schema. Using time-series data, including sensor readings and equipment statuses, participants will organise and standardise this information to streamline management processes. This effort minimises manual intervention, making smart building technologies more accessible and sustainable.

![](https://images.aicrowd.com/uploads/ckeditor/pictures/1371/Screenshot_2024-12-06_at_10.09.24.png =600x342)

# 📊 Dataset

The dataset incorporates time-series data from three anonymised buildings in Australia, including signals such as temperature readings, setpoints, and alarms from building devices. These are recorded as timestamp-value pairs with timestamps in relative terms and are characterised by irregular sampling rates.

**Data Integration and Segmentation**

*   **Data Combination and Partitioning**: All data from the three buildings are combined into a single dataset and then segmented into distinct sets for training, leaderboard testing, and secret competition testing. This partitioning deliberately varies in proportion to test the algorithms’ generalisation capabilities across buildings differently represented in the training data.
*   **Time Series Chunking**: The dataset is further divided into shorter segments or chunks with durations ranging from 2 to 8 weeks. This approach is designed to evaluate algorithm performance across various observation windows.

# 🏷️ Labeling Structure

The labelling adheres to a modified version of [Brick schema version 1.2.1](https://brick.andrew.cmu.edu/ontology/1.2/classes/Point/), featuring 94-point sub-classes. Each data point is classified with multiple label types:

*   **Positive Labels**: The true label and its parent classes.
*   **Zero Labels**: All subclasses of the true label.
*   **Negative Labels**: All unrelated labels.

A predictive model that identifies a label more specific than the true label is not penalised, promoting precision without discouragement. This flexible labelling structure aims to foster accurate and specific classifications. The dataset is distributed under the CC BY 4.0 license, ensuring open access and reuse.

![](https://images.aicrowd.com/uploads/ckeditor/pictures/1372/Screenshot_2024-12-06_at_10.09.13.png =600x307)

# 🛠️ Starter Kit and Resources

The [official repository of the NeurIPS 2024 dataset paper](https://github.com/cruiseresearchgroup/DIEF_BTS) provides comprehensive experimental results, including several baseline models. It features a selection of naive baselines that do not consider specific features, traditional machine learning baselines, and advanced deep learning baselines, one of which was enhanced through hyperparameter tuning. The repository details each model’s performance, highlighting that even the best-performing Transformer model, with tuning, shows only marginal improvement over simpler methods.

Participants looking to explore existing classification strategies can refer to the [Papers with Code](https://paperswithcode.com/task/time-series-classification) website, which catalogues techniques previously applied to similar challenges.

**📚 Make your first submission with ease [using this starter-kit](https://github.com/cruiseresearchgroup/DIEF_BTS/tree/main/competition1).**

# 📈 Evaluation Criteria

This task is classified as a multilabel timeseries classification challenge, where participants are tasked with predicting multiple labels for each timeseries data point. Challenge employs the Brick ontology version 1.2.1 for label definitions, including 94-point sub-classes found within the buildings in our dataset.

*   **Micro F1 Calculation**: The evaluation metric is the F1 score, calculated on a micro level for each label. This involves computing precision and recall for each label while masking zero labels to ignore predictions unrelated to the actual labels in the data.
*   **Label Masking**: Predictions more specific than the ground truth will not be penalised to foster a fair evaluation. The latter is still accepted if the true label is less specific than the participant’s prediction. This allows predictions that enhance specificity without the risk of penalisation for over-specification.
*   **Final Scoring**: The final score for each participant is determined by averaging the micro F1 scores across all labels. This balances the need for precision with the flexibility to be more specific. The combined score from the leaderboard and competition test sets will ultimately determine the final rankings.

# 📇 Evaluation Metric Calculation Process

## Ground Truth Label

The ground truth label is a 2D array (N, C), where:
- **N** is the number of instances in the test set,
- **C** is the number of classes,
- Each value is either **1**, **0**, or **-1**.

A **0** value means that it is not known if it belongs to that label and, therefore, is **masked**. The ground truth label for the test set is not available to you.

## Input Format

You will provide one file: **Predicted Confidence Scores (h)**:
- A 2D array (N, C),
- Each value is a floating-point number in [0, 1].

## Thresholding

We apply a 0.5 threshold to each confidence score h[i, j]:

- ≥ 0.5 → Predicted as **+1 (Positive)**
- < 0.5 → Predicted as **-1 (Negative)**

Zero-labelled ground truth entries (y[i, j] = 0) are excluded from metric calculations.

## Example with Colour-Coding

**Given:**
- N=5 samples, C=1 class
- y = [1, 0, 1, -1, 1]
- h = [0.8, 0.7, 0.9, 0.4, 0.55]

**Thresholding at 0.5:**
- 0.8 → +1
- 0.7 → +1 (masked out since ground truth = 0)
- 0.9 → +1
- 0.4 → -1
- 0.55 → +1


**Table:**

![](https://images.aicrowd.com/uploads/ckeditor/pictures/1374/tableConvert.com_gmxta3.png =600x278)

**Counts** (excluding masked sample #2):  
- TP = 3
- TN = 1
- FP = 0
- FN = 0

**Metrics:**
- Accuracy = (TP + TN) / (TP+TN+FP+FN) = (3+1)/4 = 1.0
- Precision = TP / (TP+FP) = 3/3 = 1.0
- Recall = TP / (TP+FN) = 3/3 = 1.0
- F1 = 1.0

## Mean Average Precision (mAP)

To compute **mAP**, we consider multiple thresholds, produce Precision-Recall curves, and integrate to find the Average Precision for each class. The mAP is the mean of these AP values.


# 🔍 Solution Validation and Reporting

Participants can upload up to ten submissions per day in CSV format. Each submission must adhere strictly to the prescribed format to ensure accurate leaderboard evaluations, reflecting the test set's real-time performance.

The competition is structured into two rounds, allowing participants to refine their entries if necessary. To maintain fairness and the integrity of the results, standard AIcrowd competition rules will apply:

1.  **Code and Model Validation**: The top submissions that qualify for final consideration will undergo rigorous scrutiny to validate the code and models, ensuring that only the provided dataset has been used.
2.  **Dataset Limitations**: Participants must exclusively train their models using the dataset provided by the organisers. The use of external datasets is strictly prohibited.
3.  **Solution Documentation**: Winners must document their methodology thoroughly. This includes a mandatory detailed solution report that should be submitted for publication on arXiv, fostering transparency and contributing to the community's collective knowledge.

In addition, participants must meet a minimum performance threshold to qualify for prize consideration, ensuring that only the most effective solutions are rewarded.

# 📅 Timeline

The challenge will follow the timeline outlined below. Please note that some end dates for the rounds may change; any updates will be communicated promptly.

*   **Launch of Round 1**: 9th December 2024, 23:05 AoE
*   **Launch of Round 2**: 9th January 2025, 23:05 AoE
*   **Completion of Competition**: 3rd February 2025, 23:05 AoE
*   **Winner Announcement**: 7th February 2025
*   **Presenting at The Web Conference 2025 (TBC)** 

# 🏆 Prizes

The total prize pool is $20,000 AUD. The cash prizes and travel grants are distributed to five winners.

**Cash Prizes**

*   **🥇1st Place:** 5,000 AUD
*   **🥈2nd Place:** 3,000 AUD
*   **🥉3rd Place:** 3 x 1,000 AUD

**Travel Grants**

Challenge prizes include the in-person presentation grant, which will be paid to winning teams presenting their solutions in person at the The Web Conference 2025 (TBC). This remuneration is to be paid after the presentation.

*   **🥇1st Place**: 2,500 AUD
*   **🥈2nd Place**: 2,000 AUD
*   **🥉3rd Place**: 3 x 1,500 AUD

# 🌍 Join Us in Building the Future of Smarter, Greener Buildings!

Help shape the future of building management with innovative data classification. Your contribution could redefine how we manage buildings globally, making them more efficient, sustainable, and connected. Sign up, put your skills to the test, and take part in the journey to a smarter world!