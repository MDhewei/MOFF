[![](https://img.shields.io/badge/Pypi-v1.2.3-519dd9.svg)](https://pypi.org/project/MOFF/)
[![License: GUN](https://img.shields.io/badge/License-GUN-yellow.svg)](https://github.com/MDhewei/MOFF/blob/master/LICENSE)
![](https://img.shields.io/badge/language-python-orange.svg)

## Introduction of MOFF 

<div align="center"><img src="Figures/moff-high-resolution-logo-color-on-transparent-background.png"  height="200" width="700"></div>
  
                                # # # # #   # # # #   # # # #  # # # # 
                                #   #   #   #     #   #        #       
                                #   #   #   #     #   # # # #  # # # #    
                                #   #   #   #     #   #        #          
                                #   #   #   # # # #   #        #          
                                     

    Hi, my name is MOFF. I was designed to predict off-target effects for CRISPR/Cas9! 
    
    I have three major functions: 1). Predict off-target effects for any given gRNA-target pair.
                                  2). Predict genome-wide off-target effects for any sgRNA.
                                  3). Select best sgRNAs for allele-specific knockout.
    
    
    Hope you enjoy playing with me ^o^!
                                     
    Any questions or bugs, please contact hwkobe.1027@gmail.com or whe3@mdanderson.org
                                     


## How to install MOFF


### Requirements

- Python>=3.4
- pandas, numpy, scipy
- tensorflow, keras
 
### Installation 
 Step1: Install Anaconda (highly recomended)
    
 ```console
 wget https://repo.continuum.io/archive/Anaconda2-2018.12-Linux-x86_64.sh 
 bash Anaconda2-2018.12-Linux-x86_64.sh 
 ```

 Step2: Install MOFF through pip
 ```console     
 pip install MOFF
 ```
    
 Step3: **OR** you can install MOFF through git clone
 ```console   
 git clone https://github.com/MDhewei/MOFF.git
 cd MOFF
 python setup.py install
 ```
    

## How to use MOFF

### 1. MOFF score: Predict off-target effects for given gRNA-target pairs


#### Inputs for MOFF score

MOFFscore requires the user to provide .csv or .txt file containing sgRNA sequences and corresponding 
DNA target sequences. Each line should have one gRNA(20bp+PAM) and one target(20bp+PAM) sequence. Note that MOFF is designed
for mismatch-only off-target prediction, not for indel mutations. See example file [here.](https://github.com/MDhewei/MOFF/blob/master/MOFF/MOFFscore_test.txt)

                    
     Arguments of the program:

     -i/--inputfile (required): 
     Input files containing sgRNA sequences and corresponding DNA target sequences.
 
     -p/--prefix (Optional): 
     Prefix of the file to save the outputs, default: ScoreTest.

     -o/--outputdir (Optional): 
     Directory to save output files, if no directory is given a folder named MOFF_scores
     will be generated in current working directory.
 

#### Example to run MOFFscore

```console
MOFF score -i MOFFscore_test.txt
```

#### Columns of Output table

    - crRNA: the sgRNAs designed to target specific DNA sequence
    - DNA: the DNA sequence of the off-target site 
    - MDE: predicted mismatch-dependent off-target effect 
    - CE:  predicted combinatorial effect
    - MMs: the number of mismatches between sgRNA and off-target
    - GMT: predicted guide-intrinsic mismatch tolerence 
    - MOFF: the final MOFF score predicted for given gRNA-target pair



### 2. MOFF aggregate: Predict the genome-wide off-target effects for given sgRNAs

  MOFF aggregation can directly take the outputs of [CRISPRitz](https://github.com/pinellolab/CRISPRitz) and [Cas-OFFinder](http://www.rgenome.net/cas-offinder/) as inputs. Output table files generated by any genome-wide off-target searching methods are also supported in theory, but the columns of outputs for different methods are different, therefore, it is required to modify the column name of sgRNA(20bp+PAM) and target(20bp+PAM) to 'crRNA' and 'DNA' respectively. Note that MOFF only supports mismatch-only off-target predictions, indel mutations are not applicable. File formats including .csv and .txt are accepted. See examplefile [here.](https://github.com/MDhewei/MOFF/blob/master/MOFF/MOFFaggregation_test.txt)

     Arguments of the program:

     -i/--inputfile (required): 
     Input files containing all the potential off-target sites in the genome for given sgRNA(s).

     -p/--prefix (Optional): 
     Prefix of the file to save the outputs, default: AggregationTest.

     -o/--outputdir (Optional): 
     Directory to save output files, if no directory is given a folder named MOFF_aggregation
     will be generated in current working directory.
 

#### Example to run MOFF aggregate

```console
MOFF aggregate -i MOFFaggregation_test.txt
```

#### Columns of Output table

    - sgRNA: the sgRNAs selected to predict genome-wide off-target
    - GMT: aggregated guide-intrinsic mismatch tolerence 
    - MDE.sum: aggregated mismatch-dependent off-target effect 
    - MOFF.sum: aggregated MOFF score for specific sgRNA
    
### 3. MOFF allele: Predict the genome-wide off-target effects for given sgRNAs

MOFF allele requires the users to input local DNA sequences of a wild-type allele and a mutant allele. 
The two DNA sequences should be of the same length. There should be at least one hit of 20bp+PAM(NGG) in 
the DNA sequence for knockout and the mutation point should be included within the hit. If you 
want to design the sgRNA to specifically target the WT allele, just input DNA sequence of WT as mutant 
and mutant sequence as wildtype. 

     Arguments of the program:

     -m MUTANT, --mutant MUTANT
                Local DNA sequence of mutant allele, at least one hit of 20bp (mutation sites included)
                followed by PAM (NGG) should be included, if more than one hit is found, MOFF will
                design sgRNAs based on all possible PAMs.
                
     -w WILDTYPE, --wildtype WILDTYPE
                  Local DNA sequence of wild type allele paired with the mutant allele, which should be
                  the same length as the mutant allele DNA sequence.
                 
     -p PREFIX, --prefix PREFIX
                Prefix of the file to save the outputs, in the format PREFIX + _allele_specific_knockouts, 
                default: AlleleTest.
                
     -o OUTPUTDIR, --outputdir OUTPUTDIR
                   Directory to save output files, if no directory is given, a output folder named
                   MOFF_Allele will be generated in current working directory.
                    

#### Example to run MOFF allele

    For example two mutant for DNMT3a:
    ACTGACGTCTCCAACATGAGC|CGC|TTGGCGAGGCAGAGACTGCT (WT)
    ACTGACGTCTCCAACATGAGC|tGC|TTGGCGAGGCAGAGACTGCT (R882C)
    ACTGACGTCTCCAACATGAGC|CaC|TTGGCGAGGCAGAGACTGCT (R882H)
    
    1). To knockout R882C allele
    MOFF allele -m ACTGACGTCTCCAACATGAGCTGCTTGGCGAGGCAGAGACTGCT -w ACTGACGTCTCCAACATGAGCCGCTTGGCGAGGCAGAGACTGCT -p R882C
    
    2). To knockout R882H allele
    MOFF allele -m ACTGACGTCTCCAACATGAGCCACTTGGCGAGGCAGAGACTGCT -w ACTGACGTCTCCAACATGAGCCGCTTGGCGAGGCAGAGACTGCT -p R882H
    
    3). To knockout WT in R882C cell
    MOFF allele -m ACTGACGTCTCCAACATGAGCCGCTTGGCGAGGCAGAGACTGCT -w ACTGACGTCTCCAACATGAGCTGCTTGGCGAGGCAGAGACTGCT -p WTinR882C
    
    4). To knockout WT in R882H cell
    MOFF allele -m ACTGACGTCTCCAACATGAGCCGCTTGGCGAGGCAGAGACTGCT -w ACTGACGTCTCCAACATGAGCCACTTGGCGAGGCAGAGACTGCT -p WTinR882H
    

#### Columns of Output table

    - sgRNA: all the possible sgRNAs selected for allele-specific knockouts
    - DNA_KO: DNA target of allele you want to knockout, usually it is the mutant allele
    - DNA_NA: DNA target of allele you want to keep, usually it is the wild-type allele
    - GMT: Guide-intrinsic mismatch tolerence for the designed sgRNA
    - MOFF_KO: the predicted MOFF score to target the DNA-KO
    - MOFF_NA: the predicted MOFF score to target the DNA-NA
    - MOFF_ratio: the ratio between MOFF_NA/MOFF_KO
    
**To knockout desired allele, please select sgRNA with a high MOFF_KO score to knockout (MOFF_KO > 0.5 is suggested)**

**To maintain specificity of sgRNA, please select sgRNA with low MOFF_NA relative to MOFF_KO (MOFF_ratio < 0.2 is suggested)**

