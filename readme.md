# *Freezing of Gait Detection Using Gramian Angular Fields and Federated Learning from Wearable Sensors*

***Shovito Barua Soumma, S M Raihanul Alam, Rudmila Rahman, Umme Niraj Mahi, Abdullah Mamun, Sayyed Mostafa Mostafavi, Hassan Ghasemzadeh***
---

#### `FOGSense is currently under review. The complete code and the preprocessed dataset will be made available at this repository after the acceptance of our submission. The reviewers have access to the source code in the supplementary materials of our submission.`

```
pip3 freeze > requirements.txt
```


## Table of Contents
- [To Cite This Work](#to-cite-this-work)
- [Abstract](#abstract)
- [Getting the Code](#getting-the-code)
- [Running the Code](#running-the-code)
  - [Requirements](#requirements)
  - [Setup Environment](#setup-environment)
  - [Reproducing the Results](#reproducing-the-results)




## Abstract
Freezing of gait (FOG) is a debilitating symptom of Parkinson's disease (PD) that impairs mobility and safety. Traditional detection methods face challenges due to intra and inter-patient variability, and most systems are tested in controlled settings, limiting their real-world applicability. Addressing these gaps, we present ***FOGSense***, a novel FOG detection system designed for uncontrolled, free-living conditions.![](manuscript-supplementary/method.png) It uses Gramian Angular Field (GAF) transformations and federated deep learning to capture temporal and spatial gait patterns missed by traditional methods. We evaluated our FOGSense system using a public PD dataset, 'tdcsfog'. FOGSense improves accuracy by 10.4% over a single-axis accelerometer, reduces failure points compared to multi-sensor systems, and demonstrates robustness to missing values. The federated architecture allows personalized model adaptation and efficient smartphone synchronization during off-peak hours, making it effective for long-term monitoring as symptoms evolve. Overall, FOGSense achieves a 22.2% improvement in F1-score compared to state-of-the-art methods, along with enhanced sensitivity for FOG episode detection.

## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://github.com/shovito66/FOGSense) repository:
  ```
    git clone git@github.com:shovito66/FOGSense.git
  ```
or [download a zip archive](https://github.com/shovito66/FOGSense/archive/master.zip).

# Running the code

-----
## Requirements
We use `conda` virtual environments to manage the project dependencies in
isolation.
Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).

Run the following command in the repository folder (where `main.py`
is located) to create a separate environment and install all required
dependencies in it:
    
    conda env create

[//]: # (## Reproducing the results)
## Setup Environment
Before running any code you must activate the conda environment:
    
    source activate ENVIRONMENT_NAME

or, if you're on Windows:

    activate ENVIRONMENT_NAME
**if you are on Windows/Linux:** To install the necessary dependencies, you can use the provided `requirements.txt` file. Run the following command:

    pip install -r requirements.txt
**if you are on Macos**: Use `requirements-macos.txt` file (inside `/code` directory)  to install the dependencies

    pip install -r requirements-macos.txt

----
## Reproducing the Results/ How To Run The Code
1. **Download the dataset**:

# Contact
>For any questions or issues, please contact 
*  ***[Shovito Barua Soumma](https://www.shovitobarua.com)*** at [shovito@asu.edu](shovito@asu.edu) or 
*  ***[Abdullah Mamun](https://www.abdullah-mamun.com)*** at [a.mamun@asu.edu](a.mamun@asu.edu).



[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)