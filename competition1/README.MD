![A promotional flyer for "Brick By Brick 2024: Automating Building Data Classification," featuring red pixel-like building graphics on the right. The flyer includes the UNSW Sydney logo and AICrowd logo at the top. Below the event title, it highlights a $11,000 AUD cash prize and a $9,000 AUD travel grant. The event URL is provided as "aicrowd.com/brick2024".](/competition1/2024_12_09_Flyer_by_AICrowd.jpg)

# Brick by Brick: Automating Building Data Classification Challenge

A global challenge to automate building data classification, unlocking more intelligent, energy-efficient buildings for a sustainable future.

Buildings are one of the biggest energy consumers in the modern world, making energy efficiency essential. However, managing building systems data across different buildings is time-intensive and costly due to inconsistent data formats. This challenge invites you to transform building management by creating a solution that classifies building data automatically, promoting standardised, energy-efficient management for a more sustainable world.

Check the official competition page for more information: https://www.aicrowd.com/challenges/brick-by-brick-2024

Watch the online townhall session: https://youtu.be/kYyIguY2Kso?si=PwseqtKoHpuumPSw

# Starter Kit

To get started, follow these seven easy steps:

1. Join the challenge from the official AICrowd website and accept the terms: https://www.aicrowd.com/challenges/brick-by-brick-2024
2. Download the dataset: https://www.aicrowd.com/challenges/brick-by-brick-2024/dataset_files
3. Download [`random_submission.py`](/competition1/random_submission.py)
4. Create a directory called `data/` in the same location as the `random_submission.py` file 
5. Move the downloaded dataset files into the `data/` directory. 
6. Run [`random_submission.py`](/competition1/random_submission.py) locally, and it will generate a sample submission file `data/sample_submission_v0.1.0.csv.gz`. Might take few minutes.
7. Upload a sample submission file `data/sample_submission_v0.1.0.csv.gz`

You can also use other codes in this library. Note that there are slight changes in the setup between the benchamrk for the paper and the competiton.

The evaluation function is available here: https://github.com/cruiseresearchgroup/DIEF_BTS/blob/9ac4f4ea077ef0fed129a6761d77e1563056f6c2/20240530_class_code/thuml_tslib_dief/diefComp1Utils.py#L338

# Winners

![The image shows a leaderboard from a machine learning competition ranking teams by F1 Score. The top team, yddm, has an F1 Score of 0.558, last submitted on February 4, 2025. XiaobaiLan follows with 0.544, last submitted on January 28, 2025, and NaiveBaes is third with 0.528, last submitted on February 3, 2025. The table includes team names, performance metrics, number of entries, and submission timestamps, with the top three teams highlighted in gold, silver, and bronze.](/competition1/20250205_leaderboard.png)

Congratulation to the top 5 winners! Read the official announcement on [AIcrowd](https://discourse.aicrowd.com/t/brick-by-brick-challenge-2024-winner-announcement-their-solutions/16878).

The documentations to the winning solutions are published in [WWW '25: Companion Proceedings of the ACM on Web Conference 2025](https://dl.acm.org/doi/proceedings/10.1145/3701716#heading37):
1. 
Chengfeng Qiu, Jiahui Zhou, Yongfeng Liao, Zhengliang Cui, and Dan Li. 2025.
[A Unified CatBoost Framework for IoT Data Classification in Buildings Using Multi-Level Feature Extraction](https://dl.acm.org/doi/10.1145/3701716.3718479).
In Companion Proceedings of the ACM on Web Conference 2025 (WWW '25). Association for Computing Machinery, New York, NY, USA, 3026–3029. https://doi.org/10.1145/3701716.3718479
2. [Winning Brick by Brick with Daily Slices: A 94-Task Unified XGBoost Solution for Brick Schema Classification](https://dl.acm.org/doi/10.1145/3701716.3718482)
3. [Hierarchical Multi-Label Classification of Building Management System Time-Series Data](https://dl.acm.org/doi/10.1145/3701716.3718484)
4. [Another Brick in the Wall: Leveraging Feature Extraction and Ensemble Learning for Building Data Classification](https://dl.acm.org/doi/10.1145/3701716.3718480)
5. [BrickMIR: A Minimal, Imbalance-tuned, and Ratio-based Framework for Brick Metadata Classification](https://dl.acm.org/doi/10.1145/3701716.3718481)

The winner's code will be linked from here: (last update 2025 05 29)

# Post-competition

The training and testing files, as well as some auxilary files, are archived and made publicly available in prepetuity on [FigShare](https://figshare.com/articles/dataset/_AIcrowd_Brick_by_Brick_2024_dataset/28720391).
`DOI: 10.6084/m9.figshare.28720391`

# Competition 2: FlexTrack: Detecting Energy Flexibility in Buildings

![Promotional graphic for “FlexTrack: Detecting Energy Flexibility in Buildings,” announcing a 2025 launch with a QR code, contact details, and logos of supporting organizations.](/competition1/comp2Promo.png)

Register your interest here: https://forms.office.com/r/9zWyNULv8j

Contact: Dr. Emily Yap (eyap@uow.edu.au)

# Cite

Arian Prabowo, Xiachong Lin, Imran Razzak, Hao Xue, Matthew Amos, Stephen D. White, and Flora D. Salim. 2025.
[Brick-by-Brick: Cyber-Physical Building Data Classification Challenge](https://dl.acm.org/doi/10.1145/3701716.3718483).
In Companion Proceedings of the ACM on Web Conference 2025 (WWW '25). Association for Computing Machinery, New York, NY, USA, 3021–3025. https://doi.org/10.1145/3701716.3718483

```
@inproceedings{prabowo2025BrickByBrick,
    author = {Prabowo, Arian and Lin, Xiachong and Razzak, Imran and Xue, Hao and Amos, Matthew and White, Stephen D. and Salim, Flora D.},
    title = {Brick-by-Brick: Cyber-Physical Building Data Classification Challenge},
    year = {2025},
    isbn = {9798400713316},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3701716.3718483},
    doi = {10.1145/3701716.3718483},
    booktitle = {Companion Proceedings of the ACM on Web Conference 2025},
    pages = {3021–3025},
    numpages = {5},
    keywords = {building, classification, machine learning, ontology, timeseries},
    location = {Sydney NSW, Australia},
    series = {WWW '25}
}
```

# License
The intellectual property of the following items are owned by [UNSW Sydney](https://www.unsw.edu.au/):
* This starter kit
* [This graphical asset](/competition1/2024_12_09_Flyer_by_AICrowd.jpg)
* [The challenge description](/competition1/updated_overview.md)
