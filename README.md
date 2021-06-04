## Introduction to MOFF                                   
                           # # # # #   # # # #   # # # #  # # # # 
                           #   #   #   #     #   #        #       
                           #   #   #   #     #   # # # #  # # # #    
                           #   #   #   #     #   #        #          
                           #   #   #   # # # #   #        #          
                                     

    Hi,My name is MOFF, I was designed to predict off-target effects for CRISPR/Cas9! 
    
    I have three major functions: 1). Predict off-target effects for any given gRNA-target pair.
                                  2). Predict genome-wide off-target effects for any sgRNA.
                                  3). Select best sgRNAs for allele-specific knockout.
    
    
    Hope you enjoy playing with me ^o^!
                                     
    Any questions or bugs, please concat through hwkobe.1027@gmail.com or whe3@mdanderson.org
                                     


## How to install MOFF?

 **Note: ProTiler is written in Python,Python>=2.7 is needed**

    Step1: Install Anaconda (highly recomended)

    ```console
    wget https://repo.continuum.io/archive/Anaconda2-2018.12-Linux-x86_64.sh 
    bash Anaconda2-2018.12-Linux-x86_64.sh 
    ```

    Step2: Install required python packages

    ```console
    pip install matplotlib==2.2.3 pandas sklearn numpy seaborn
    ```

    Step3: Install MOFF through pip
    ```console
    pip install MOFF
    ```

    Step4: OR you can nstall MOFF through git clone
    ```console
    git clone https://github.com/MDhewei/MOFF.git
    cd MOFF
    python setup.py install
    ```


## How to use MOFF? 

### 1. MOFF score: Predict off-target effects for given gRNA-target pairs

#### Inputs for MOFF score

     MOFFscore require user to provide .csv or .txt file containing sgRNA sequences and corresponding 
     DNA target sequences.One gRNA(20bp+PAM) and one target(20bp+PAM) per line. Note that MOFF is designed
     for mismatch-only off-target prediction, not for indel mutations.See the following example:

                        GAGTCCGAGCAGAAGAAGAATGG,GAGTCCAAGTAGAAGAAAAATGG
                        GTTGCCCCACAGGGCAGTAAAGG,GTGGACACCCCGGGCAGGAAAGG
                        GGGTGGGGGGAGTTTGCTCCAGG,AGGTGGGGTGAGTTTGCTCCAGG


     Arguments of the program:

     -i/--inputfile (required): 
     Input files containing sgRNA sequences and corresponding DNA target sequences.

     -p/--prefix (Optional): 
     Prefix of the file to save the outputs,default: ScoreTest.

     -o/--outputdir (Optional): 
     Directory to save output files,if no directory is given a folder named MOFF_scores
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

#### Inputs for MOFF aggregate

     MOFF aggregation can directly take the outputs of CRISPRitz as inputs. Besides, output table files
     generated by any genome-wide off-target searching methods such as Cas-OFFinder are supported in theory,
     but the columns of outputs for different methods are different, thus it is required to modify the column
     name of sgRNA(20bp+PAM) and target(20bp+PAM) to 'crRNA' and 'DNA' respectively. Note that MOFF only
     support mismatch-only off-target predictions, indel mutations are not applicable.File formats including
     .csv and .txt are accepted.

     Arguments of the program:

     -i/--inputfile (required): 
     Input files containing all the potneital off-target sites in the genome for given sgRNA(s)

     -p/--prefix (Optional): 
     Prefix of the file to save the outputs,default: AggregationTest.

     -o/--outputdir (Optional): 
     Directory to save output files,if no directory is given a folder named MOFF_aggregation
     will be generated in current working directory.
 

#### Example to run MOFF aggregate

```console
   MOFF aggregate -i MOFFaggregation_test.txt
```

#### Columns of Output table

    - sgRNA: the sgRNAs selected to predict genome-wide off-target
    - MDE.sum: aggregated mismatch-dependent off-target effect 
    - GMT.sum: aggregated guide-intrinsic mismatch tolerence 
    - MOFF.sum: aggregated MOFF score for specific sgRNA


### 3. MOFF allele: Predict the genome-wide off-target effects for given sgRNAs

#### Inputs for MOFF allele

     MOFF allele require the users to input two local DNA sequences of wild-type allele and mutant 
     allele. Two DNA sequence should be of same length. There should be at least one hit of 20bp+PAM(NGG) 
     in the DNA sequence to be knockout and the mutation point should be included within the hit.
     
**Note that, if you want to design sgRNA specifically target WT allele, you just input DNA sequence of 
WT as mutant and mutant sequence as wildtype**

     Arguments of the program:

     -m MUTANT, --mutant MUTANT
                Local DNA sequence of mutant allele, at least one hit of 20bp(mutation sites included)
                followed by PAM (NGG) should be included, if more than one hits found, MOFF will
                design sgRNAs based on all possible PAMs.
                
     -w WILDTYPE, --wildtype WILDTYPE
                  Local DNA sequence of wild type allele paired with the mutant allele,which should be
                  the same length of the mutant allele DNA sequence.
                 
     -p PREFIX, --prefix PREFIX
                Prefix of the file to save the outputs, default: AlleleTest.
                
     -o OUTPUTDIR, --outputdir OUTPUTDIR
                    Directory to save output files,if no directory is given, a output folder named
                    MOFF_aggregation will be generated in current working directory.
                    

 

#### Example to run MOFF allele

```console
   python MOFF.py allele -m ACTGACGTCTCCAACATGAGCTGCTTGGCGAGGCAGAGACTGCT -w ACTGACGTCTCCAACATGAGCCGCTTGGCGA -p R882C
```

#### Columns of Output table

    - sgRNA: all the possible sgRNAs selected for allele-specific knockouts
    - DNA_KO: DNA target of allele you want to knockout, usually it is the mutant allele
    - DNA_NA: DNA target of allele you want to keep, usually it is the wild-type allele
    - MOFF_KO: the predicted MOFF score to target the DNA-KO.
    - MOFF_NA: the predicted MOFF score to target the DNA-NA.
    
**It is practical to select sgRNA with high MOFF score to knockout allele but low MOFF score of non-knockout allele,
so that sgRNA can specifically knockout the desried allele.**
