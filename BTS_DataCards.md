# Building Timeseries (BTS)
The Building TimeSeries (BTS) dataset covers three buildings over a three-year period, comprising more than ten thousand timeseries data points with hundreds of unique ontologies.
Moreover, the metadata is standardized using the Brick schema.

Buildings play a crucial role in human well-being, influencing occupant comfort, health, and safety.
Additionally, they contribute significantly to global energy consumption, accounting for one-third of total energy usage, and carbon emissions.
Optimizing building performance presents a vital opportunity to combat climate change and promote human flourishing.
However, research in building analytics has been hampered by the lack of accessible, available, and comprehensive real-world datasets on multiple building operations.
To demonstrate the utility of this dataset, we performed benchmarks on two tasks: timeseries ontology classification and zero-shot forecasting.
These tasks represent an essential initial step in addressing challenges related to interoperability in building analytics.

(last update of this .md file: 12 06 2024)

#### Dataset Link
https://github.com/cruiseresearchgroup/DIEF_BTS/

#### Data Card Author(s)

- **Arian Prabowo, UNSW:** (Contributor)

## Authorship
### Dataset Owners
#### Team(s)
[CSIRO, Energy](https://www.csiro.au/en/research/technology-space/energy)
and
[CRUISE research group](https://cruiseresearchgroup.github.io/)

#### Contact Detail(s)
- **Dataset Owner(s):** Arian Prabowo, Matthew Amos, and Flora D. Salim
- **Affiliation:** UNSW, CSIRO
- **Contact:** arian.prabow@unsw.edu.au, matt.amos@csiro.au, flora.salim@unsw.edu.au
- **Website:** [CSIRO, Energy](https://www.csiro.au/en/research/technology-space/energy) and [CRUISE research group](https://cruiseresearchgroup.github.io/)

#### Author(s)
- Arian Prabowo, UNSW
- Xiachong Lin, UNSW
- Imran Razzak, UNSW
- Hao Xue, UNSW
- Emily W. Yap, UOW
- Matthew Amos, CSIRO
- Flora D. Salim, UNSW

### Funding Sources
This is a part of NSW DIEF project: https://research.csiro.au/dch/projects/nsw-dief/

#### Institution(s)
TBA

#### Funding or Grant Summary(ies)
<!-- scope: periscope -->
<!-- width: full -->
<!-- info: Provide a short summary of programs or projects that may have funded
the creation, collection, or curation of the dataset.

Use additional notes to capture any other relevant information or
considerations. -->
<!-- *For example, Institution 1 and institution 2 jointly funded this dataset as a
part of the XYZ data program, funded by XYZ grant awarded by institution 3 for
the years YYYY-YYYY.*

Summarize here. Link to documents if available. -->
TBA

<!-- **Additional Notes:** Add here -->

## Dataset Overview
#### Data Subject(s)
<!-- scope: telescope -->
<!-- info: Select ***all applicable**** subjects contained the dataset: -->
- Non-Sensitive Data about people
- Data about natural phenomena
- Data about places and objects
- Data about systems or products and their behaviors

#### Dataset Snapshot
<!-- scope: periscope -->
<!-- info: Provide a snapshot of the dataset:<br><br>(Use the additional notes
to include relevant information, considerations, and links to table(s) with
more detailed breakdowns.) -->

Category | Data
--- | ---
Number of Buildings| 3
Size of Dataset | 18.77  GB
Number of Datapoint | 2 863 795 583
Number of Timeseries | 14 547
Number of Unqiue Brick Class of the Timeseries | 215
Start Date | 01-01-2021
End Date| 18-01-2024
Duration | 1 112 days

**Above:** Summary statistics of the timeseries.

Category | BTS_A | BTS_B | BTS_C
--- | --- | --- | ---
Collection | 4    (2)   | 2   (2)  | 8     (1)  
Equipment  | 547  (24)  | 159 (25) | 963   (41) 
Location   | 481  (9)   | 68  (17) | 381   (26) 
Point      | 8374 (126) | 851 (57) | 10440 (159) 
Alarm      | 798  (16)  | 5   (2)  | 109   (8)  
Command    | 363  (6)   | 97  (5)  | 785   (13) 
Parameter  | 79   (6)   | 36  (2)  | 935   (17) 
Sensor     | 4396 (56)  | 266 (25) | 4062  (68) 
Setpoint   | 772  (26)  | 232 (16) | 1629  (41) 
Status     | 1628 (17)  | 110 (6)  | 2187  (19) 


**Above:** Summary statistics of the Brick Schema Metadata. (Bracketed numbers are the number of unique instances).

<!-- **Additional Notes:** Add here. -->

#### Content Description
<!-- scope: microscope -->
<!-- info: Provide a short description of the content in a data point: -->
Each datapoint in a timeseries is a pair of timestamp and value.
A timeseries is a series of datapoints, and it has an associated StreamID.
All the metadata about the StreamID are available in the metadata files.
The metadata files follow the [Brick Schema](https://brickschema.org/ontology/1.2/) and formatted as a [Turtle](https://www.w3.org/TR/turtle/) .ttl file.

<!-- **Additional Notes:** Add here. -->

#### Descriptive Statistics
<!-- width: full -->
<!-- info: Provide basic descriptive statistics for each field.

Use additional notes to capture any other relevant information or
considerations.

Usage Note: Some statistics will be relevant for numeric data, for not for
strings. -->

<!-- Statistic | Field Name | Field Name | Field Name | Field Name | Field Name | Field Name
--- | --- | --- | --- | --- | --- | ---
count |
mean |
std |
min |
25% |
50% |
75% |
max |
mode |

**Above:** Provide a caption for the above table or visualization. -->

**Additional Notes:** Not applicable as there are 14 547 fields (timeseries)

### Sensitivity of Data
The dataset does not contain personally identifiable information.

### Dataset Version and Maintenance
#### Maintenance Status
<!-- scope: telescope -->
<!-- info: Select **one:** -->
**Regularly Updated** - The full version of the dataset will be made available
after the competition is completed
and the embargo lifted
and all the data are released.
No information about the upcoming competition is available yet as it is still in the planning stage.

## Example of Data Points
#### Primary Data Modality
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Time Series

#### Sampling of Data Points
<!-- scope: periscope -->
<!-- info: Provide link(s) to data points or exploratory demos: -->
- Demo Link: Check the snippet in https://github.com/cruiseresearchgroup/DIEF_BTS/

#### Data Fields
<!-- scope: microscope -->
<!-- info: List the fields in data points and their descriptions.

(Usage Note: Describe each field in a data point. Optionally use this to show
the example.) -->

Field Name | Field Value | Description
--- | --- | ---
t | Numpy array of Timestamp | Timestamp
v | Numpy array of Float | Field Value
y | String | [Brick Class](https://brickschema.org/ontology/1.2)
StreamID | String | UUID to link to the metadata.

<!-- **Above:** Provide a caption for the above table or visualization if used.

**Additional Notes:** Add here -->

#### Typical Data Point
<!-- width: half -->
<!-- info: Provide an example of a typical data point and describe what makes
it typical.

**Use additional notes to capture any other relevant information or
considerations.** -->
This is the string representation of a timeseries in the snippet.
Each timeseries is a Python dictionary with 4 items.

```
{'t': array(['2021-01-01T00:03:16.305000000', '2021-01-01T00:13:44.899000000',
        '2021-01-01T00:23:16.203000000', ...,
        '2021-08-01T20:45:03.994000000', '2021-08-01T20:55:06.504000000',
        '2021-08-01T21:05:05.066000000'], dtype='datetime64[ns]'),
 'v': array([18.8, 18.8, 18.2, ..., 32. , 32. , 32. ]),
 'y': 'Max_Temperature_Setpoint_Limit',
 'StreamID': '213ac15b_3fbd_40b7_b59b_43ab87a09260'}
```

#### Atypical Data Point
**Additional Notes:** N/A








## Motivations & Intentions
### Motivations
#### Purpose(s)
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Research

#### Domain(s) of Application
<!-- scope: periscope -->
<!-- info: Provide a list of key domains of application that the dataset has
been designed for:<br><br>(Usage Note: Use comma-separated keywords.) -->
`Timeseries Analysis`, `Buildings`, `Knowledge Graph`, `Spatiotemporal`, `Energy Use`.

#### Motivating Factor(s)
<!-- scope: microscope -->
<!-- info: List the primary motivations for creating or curating this dataset:

(Usage Note: use this to describe the problem space and corresponding
motivations for the dataset.) -->
**Importance of building analytics.**
Building analytics, also known as data-driven smart building, 
involves the automated adjustment of building operations
to minimize emissions and costs, optimize energy usage, and enhance indoor environmental quality and occupant experience,
including comfort, health, and safety.
This is particularly crucial given that buildings account for a third of global energy usage and a quarter of global carbon emissions, comparable to the transport sector.
Optimizing building performance has the potential to significantly mitigate climate change and promote human well-being.

**Literature gaps.**
This dataset addresses two critical gaps in building analytics research.
Firstly, the scarcity of publicly available and freely accessible datasets on comprehensive real-world building operations
This limitation underscores the need for datasets covering multiple buildings to address the second gap: interoperability in building analytical models.
Interoperability is crucial for scalability, allowing models to be applied across diverse buildings with differing characteristics such as climate, usage, size, regulations, budget, and architecture.
Additionally, such datasets inherently possess properties of interest to machine learning research, such as domain shift, multimodality, imbalance, and long-tailedness.

### Intended Use
#### Dataset Use(s)
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Safe for research use

#### Suitable Use Case(s)
<!-- scope: periscope -->
<!-- info: Summarize known suitable and intended use cases of this dataset.

Use additional notes to capture any specific patterns that readers should
look out for, or other relevant information or considerations. -->
Building analytics and data-driven smart buildings. Read more about this on [IEA EBC Annex81](https://annex81.iea-ebc.org/).

#### Unsuitable Use Case(s)
<!-- scope: microscope -->
<!-- info: Summarize known unsuitable and unintended use cases of this dataset.

Use additional notes to capture any specific patterns that readers should look
out for, or other relevant information or considerations. -->
For production, especially for buildings with widely different behaviour e.g. not located in Australia.

#### Research and Problem Space(s)
<!-- scope: periscope -->
<!-- info: Provide a description of the specific problem space that this
dataset intends to address. -->
Building analytics and data-driven smart buildings. Read more about this on [IEA EBC Annex81](https://annex81.iea-ebc.org/).

#### Citation Guidelines
<!-- scope: microscope -->
<!-- info: Provide guidelines and steps for citing this dataset in research
and/or production.

Use additional notes to capture any specific patterns that readers should look
out for, or other relevant information or considerations. -->
<!-- **Guidelines & Steps:** Summarize here. Include links where necessary. -->

Will be made available when the paper is published.

<!-- **BiBTeX:**
```
@article{kuznetsova2020open,
  title={The open images dataset v4},
  author={Kuznetsova, Alina and Rom, Hassan and Alldrin, and others},
  journal={International Journal of Computer Vision},
  volume={128},
  number={7},
  pages={1956--1981},
  year={2020},
  publisher={Springer}
}
``` -->

<!-- **Additional Notes:** Add here -->









## Access, Rentention, & Wipeout
### Access
#### Access Type
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- External - Open Access

#### Documentation Link(s)
<!-- scope: periscope -->
<!-- info: Provide links that describe documentation to access this
dataset: -->
- Dataset URL to the https://figshare.com/ repository will be made available after the competition.
- GitHub URL https://github.com/cruiseresearchgroup/DIEF_BTS/

#### Prerequisite(s)
<!-- scope: microscope -->
<!-- info: Please describe any required training or prerequisites to access
this dataset. -->
None.

#### Policy Link(s)
<!-- scope: periscope -->
<!-- info: Provide a link to the access policy: -->
https://help.figshare.com/article/data-access-policy

#### Access Control List(s)
<!-- scope: microscope -->
<!-- info: List and summarize any access control lists associated with this
dataset. Include links where necessary.

Use additional notes to capture any other information relevant to accessing
the dataset. -->
None

### Retention
#### Duration
<!-- scope: periscope -->
<!-- info: Specify the duration for which this dataset can be retained: -->
Indefinite

#### Policy Summary
<!-- scope: microscope -->
<!-- info: Summarize the retention policy for this dataset. -->
**Summary:** The dataset will be hosted on https://figshare.com/ and retained according to their [policy](https://figshare.com/terms).

### Wipeout and Deletion
N/A
















## Provenance
### Collection
#### Method(s) Used
<!-- scope: telescope -->
<!-- info: Select **all applicable** methods used to collect data: -->
- Telemetry

#### Methodology Detail(s)
<!-- scope: periscope -->
<!-- info: Provide a description of each collection method used.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for collection method
type.) -->

**Source:** Senaps https://products.csiro.au/senaps/

**Is this source considered sensitive or high-risk?** [Yes]

**Dates of Collection:** [01 2018 - 01 2024]

**Primary modality of collection data:**

- Time Series

**Update Frequency for collected data:**

- Static: Data was collected once from the source.





**Source:** DCH https://research.csiro.au/dch/

**Is this source considered sensitive or high-risk?** [Yes]

**Dates of Collection:** [01 2018 - 01 2024]

**Primary modality of collection data:**

- Graph Data

**Update Frequency for collected data:**

- Static: Data was collected once from the source.



#### Source Description(s)
<!-- scope: microscope -->
<!-- info: Provide a description of each upstream source of data.

Use additional notes to capture any other relevant information or
considerations. -->
- **Senaps:** Senaps https://products.csiro.au/senaps/. From the website: Senaps is an Internet of Things (IoT) Application Enablement and Data Management cloud-based platform developed and being commercialised by CSIRO’s Data61 Distributed Sensing Systems Group. Senaps is a framework which allows you to build your own product by getting data in, analysing and distributing it to custom user-facing applications. Built-in security, data storage and APIs are allowing companies in agriculture, environment, smart buildings and more, to focus on their competitive advantage. With a basic generic user interface, Senaps combines multiple datasets in a cloud environment with open APIs, allowing users to draw useful insights from data.
- **DCH:** DCH https://research.csiro.au/dch/. From the website: CSIRO’s Data Clearing House (DCH) is a cloud-based digital platform for housing, managing and extracting valuable insights from smart building data. Allowing data ingestion from a variety of sources, the DCH stores this data in an open format allowing for interoperability and data discovery.


#### Collection Cadence
<!-- scope: telescope -->
<!-- info: Select **all applicable**: -->
**Static:** Data was collected once from DCH and Senaps.

#### Data Integration
<!-- scope: periscope -->
<!-- info: List all fields collected from different sources, and specify if
they were included or excluded from the dataset.

Use additional notes to
capture any other relevant information or considerations.

(Usage Note: Duplicate and complete the following for each upstream
source.) -->
A semantic model of the building was created using DCH platform tooling.
This created Brick schema class definitions (version 1.2.1) for points within the model, and linked these points to the timeseries data ingested via MQTTS. 

#### Data Processing
<!-- scope: microscope -->
<!-- info: Summarize how data from different sources or methods aggregated,
processed, or connected.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the following for each source OR
collection method.) -->
This dataset is comprised of data collected onto CSIRO's Data Clearing House (DCH) digital platform . Connecting to the Building Management Systems (BMS), timeseries data is collected from sensors, power, water and gas meters, and other devices within the buildings and uploaded using Message Queuing Telemetry Transport Secured (MQTTS). A semantic model of the building was created using DCH platform tooling. This created Brick schema class definitions (version 1.2.1) for points within the model, and linked these points to the timeseries data ingested via MQTTS. 

Identifiers for both the point within the model, and the timeseries identifier were anonymised by generating Universally Unique Identifiers (UUID), and a three-year-period subset of the timeseries data was extracted from the DCH platform to produce this dataset. The data was not cleaned in effort to allow evaluation of various different cleaning algorithm, and to allow the evaluations of algorithms against data with realistic errors.

### Collection Criteria
#### Data Selection
<!-- scope: telescope -->
<!-- info: Summarize the data selection criteria.

Use additional notes to capture any other relevant information or
considerations. -->
Comprehensive data from 3 buildings.

#### Data Inclusion
<!-- scope: periscope -->
<!-- info: Summarize the data inclusion criteria.

Use additional notes to capture any other relevant information or
considerations. -->
Comprehensive data from 3 buildings.

#### Data Exclusion
<!-- scope: microscope -->
<!-- info: Summarize the data exclusion criteria.

Use additional notes to capture any other relevant information or
considerations. -->

Some information have been excluded for anonymisation purposes.

### Relationship to Source
#### Use & Utility(ies)
<!-- scope: telescope -->
<!-- info: Describe how the resulting dataset is aligned with the purposes,
motivations, or intended use of the upstream source(s).

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each source type.) -->
- **Building Management Systems:** To manage building operations.

#### Benefit and Value(s)
<!-- scope: periscope -->
<!-- info: Summarize the benefits of the resulting dataset to its consumers,
compared to the upstream source(s).

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each source type.) -->
- **Building Management Systems:** To manage building operations.

#### Limitation(s) and Trade-Off(s)
<!-- scope: microscope -->
<!-- info: What are the limitations of the resulting dataset to its consumers,
compared to the upstream source(s)?

Break down by source type.<br><br>(Usage Note: Duplicate and complete the
following for each source type.) -->
- The dataset has been anonymised.


#### Changes on Update(s)
<!-- scope: microscope -->
<!-- info: Summarize the changes that occur when the dataset is refreshed.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each source type.) -->
N/A. There are no plan to update this dataset after the full release after the competition.












## Human and Other Sensitive Attributes

The dataset does not contain personally identifiable information.











## Extended Use
### Use with Other Data
#### Safety Level
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Unkown

#### Limitation(s) and Recommendation(s)
<!-- scope: microscope -->
<!-- info: Fill this out if you selected "Conditionally safe to use with
other datasets" or "Should not be used with
other datasets":

Summarize limitations of the dataset that introduce foreseeable risks when the
dataset is conjoined with other datasets.

Use additional notes to capture any other relevant information or
considerations. -->

LBNL59: A similar dataset collected from Lawrence Berkeley National Laboratory Building 59

Hong, Tianzhen; Luo, Na; Blum, David; Wang, Zhe (2022). A three-year building operational performance dataset for informing energy efficiency [Dataset]. Dryad. https://doi.org/10.7941/D1N33Q

### Forking & Sampling
#### Safety Level
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Unknown

### Use in ML or AI Systems
#### Dataset Use(s)
<!-- scope: telescope -->
<!-- info: Select **all applicable** -->
- Training
- Testing
- Validation

#### Usage Guideline(s)
<!-- scope: microscope -->
<!-- info: Summarize usage guidelines or policies that consumers should be
aware of.

Use additional notes to capture any other relevant information or
considerations. -->
The dataset is sourced from only three buildings in Australia, limiting its geographical diversity. Consequently, models trained on this dataset may not generalize well to buildings in other regions with different climates, regulations, and building practices. This limitation implies that models should primarily be used for research purposes rather than direct deployment.






## Transformations
<!-- info: Fill this section if any transformations were applied in the
creation of your dataset. -->
### Synopsis
#### Transformation(s) Applied
<!-- scope: telescope -->
<!-- info: Select **all applicable** transformations
that were applied to the dataset. -->
- None



















## Annotations & Labeling
<!-- info: Fill this section if any human or algorithmic annotation tasks were
performed in the creation of your dataset. -->
Engineers constructed the brick schema for each building.

#### Annotation Workforce Type
<!-- scope: telescope -->
<!-- info: Select **all applicable** annotation
workforce types or methods used
to annotate the dataset: -->
- Human Annotations (Expert)

#### Annotation Characteristic(s)
<!-- scope: periscope -->
<!-- info: Describe relevant characteristics of annotations
as indicated. For quality metrics, consider
including accuracy, consensus accuracy, IRR,
XRR at the appropriate granularity (e.g. across
dataset, by annotator, by annotation, etc.).

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each annotation type.) -->
**Annotation Type** | **Number**
--- | ---
Number of buildings annotated | 3
<!-- 
**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here -->

#### Annotation Description(s)
<!-- scope: microscope -->
<!-- info: Provide descriptions of the annotations
applied to the dataset. Include links
and indicate platforms, tools or libraries
used wherever possible.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each annotation
type.) -->
**(Annotation Type)**

**Description:** The Brick metadata for each buildings are made using tools on the DCH platform.

**Link:** [Relevant URL link.](https://research.csiro.au/dch/)

**Platforms, tools, or libraries:**

- DCH

## Validation Types
<!-- info: Fill this section if the data in the dataset was validated during
or after the creation of your dataset. -->
#### Method(s)
<!-- scope: telescope -->
<!-- info: Select **all applicable**: -->
- Not Validated

## Sampling Methods
<!-- info: Fill out the following block if your dataset employs any sampling
methods. -->
#### Method(s) Used
<!-- scope: telescope -->
<!-- info: Select **all applicable** methods used in the creation of this
dataset: -->
- Unsampled


## Known Applications & Benchmarks
<!-- info: Fill out the following section if your dataset was primarily
created for use in AI or ML system(s) -->
The Benchmark paper is still under review.
#### ML Application(s)
<!-- scope: telescope -->
<!-- info: Provide a list of key ML tasks
that the dataset has been
used for.

Usage Note: Use comma-separated keywords. -->
Timeseries Ontology Multi-label Classification,
Zero-shot Forecasting.

## Terms of Art
### Concepts and Definitions referenced in this Data Card
<!-- info: Use this space to include the expansions and definitions of any
acronyms, concepts, or terms of art used across the Data Card.
Use standard definitions where possible. Include the source of the definition
where indicated. If you are using an interpretation,
adaptation, or modification of the standard definition for the purposes of your
Data Card or dataset, include your interpretation as well. -->
#### Timeseries
Definition: A series of time and value pair

#### Brick Schema
Definition (from the website): Brick is an open-source effort to standardize semantic descriptions of the physical, logical and virtual assets in buildings and the relationships between them. Brick consists of an extensible dictionary of terms and concepts in and around buildings, a set of relationships for linking and composing concepts together, and a flexible data model permitting seamless integration of Brick with existing tools and databases. Through the use of powerful Semantic Web technology, Brick can describe the broad set of idiosyncratic and custom features, assets and subsystems found across the building stock in a consistent matter.

Source: https://brickschema.org/

Interpretation: Also can be interpreted as knowledge graph.

#### Turtle File
Definition (from the website): A Turtle document is a textual representations of an RDF graph.

Source: https://www.w3.org/TR/turtle/

## Reflections on Data
<!-- info: Use this space to include any additional information about the
dataset that has not been captured by the Data Card. For example,
does the dataset contain data that might be offensive, insulting, threatening,
or might otherwise cause anxiety? If so, please contact the appropriate parties
to mitigate any risks. -->
No additional information.
