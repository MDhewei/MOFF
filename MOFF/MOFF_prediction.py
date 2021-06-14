#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Modular prediction of off-target effects
@author: Wei He
@E-mail: whe3@mdanderson.org
@Date: 05/31/2021
"""

#################### Import all the packages #####################
from __future__ import division                
import pandas as pd
import math,json,os,pkg_resources  
import seaborn as sns
import os,sys,logging,subprocess
import numpy as np
from itertools import product
from itertools import combinations
from scipy.stats import gmean


RequiredFilePath = pkg_resources.resource_filename(__name__, 'StaticFiles')


#################################################################


'''This function is to encode the sgRNA sequences into (16,19) vector with 0,1 presentation of 
   certain dinucleotide at certain position.

   o input:  sg_ls: A list of sgRNAs for one-hot encoding
             
   o Output: The function will return a numpy array of feature matrix for machine learning.
'''
def OneHotEndocing(sg_ls):
    di_ls = [s[0]+s[1] for s in list(product('ATGC', repeat=2))] ## Get all possible di-nucleotide combinations
    di_dic = {}
    for i in range(len(di_ls)): 
        di_dic[di_ls[i]] = i   ## Record the index of each dinucleotide

    ls_all = []   ## Initialize a list to add vector for different sgRNA
    for sg in sg_ls:
        vec_all = []
        for i in range(len(sg)-1):
            vec = [0]*len(di_ls) ## Make all position to be 0
            di = sg[i:i+2] 
            vec[di_dic[di]]=1  ## Assign 1 if certain dinucleotide appear at certain position
            vec_all.append(vec)
        
        ls_all.append(np.array(vec_all).T)
    
    return np.array(ls_all)


'''This function is obtain the mismatches between sgRNA and the target

   o input:  1). s1: the sequence of sgRNA; 2). s2: the sequence of target DNA
             
   o Output: The function will return: 1). A list of positions where mismatch happen. 
                                       2). A list of mismatch types at each position.
'''
def GetMutType(s1,s2):
    pos_ls = []; mut_ls = []
    for i in range(20): ## Go through the index along the 20bp sequence
        r = ''; d = ''
        if s1[i]!=s2[i]:
            pos = 20-i ## The index relative to PAM
            if s1[i] == 'T': r = 'U' ## Replace 'T' with 'U' in sgRNA.
            else: r = s1[i]
            ## Get mutation type given the nt at sgRNA and target
            if s2[i] == 'A': d='T' 
            elif s2[i] == 'T': d='A'
            elif s2[i] == 'C': d = 'G'
            elif s2[i] == 'G': d = 'C'
            elif s2[i] == 'N': d = s1[i]
            pos_ls.append(pos)
            mut_ls.append('p'+str(pos)+'r'+r+'d'+d) ## p3rAdC: mismatch A-G at index 3 to PAM
    return pos_ls,mut_ls


'''This function is Calculate the off-target effect by multiplying the MDE at each position

   o input:1). m1_dic: Python dic contains MDE of all the possible nucleotide mismatches (12) 
               at all possible positions (20) 
           2). sg_ls: A list of sgRNAs
           3). tg_ls: A list of DNA targets
             
   o Output: A list of calculated mismatch-dependent effect.
'''
def Multiply(m1_dic,sg_ls,tg_ls):
    me_ls = []
    for i in range(len(sg_ls)):
        s1 = sg_ls[i][0:20].upper()
        s2 = tg_ls[i][0:20].upper()
        #print (s1,s2)
        mut_ls = GetMutType(s1,s2)[1]
        score = 1
        for mut in mut_ls:  ##Multiply all the 1-mismatch effects
            score = score*m1_dic[mut] ##m1_dic: dic file 
        me_ls.append(score)
        
    return me_ls


'''This function is to get mismatch numbers of gRNA-target pairs'''

def MisNum(sg_ls,tg_ls):
    num_ls = []
    for i in range(len(sg_ls)):
        s1 = sg_ls[i][0:20].upper()
        s2 = tg_ls[i][0:20].upper()
        
        num = len(GetMutType(s1,s2)[0])
        num_ls.append(num)
        
    return num_ls


def MisType(sg_ls,tg_ls):
    tp_ls = []
    for i in range(len(sg_ls)):
        s1 = sg_ls[i][0:20].upper()
        s2 = tg_ls[i][0:20].upper()
        
        tp = '|'.join(GetMutType(s1,s2)[1])
        tp_ls.append(tp)
        
    return tp_ls


'''This function is Calculate Combinatorial effect (CE) for given mismatch positions

   o input:1). m2_dic: Python dic contains CE of all the possible position combinaitons
           2). sg_ls: A list of sgRNAs
           3). tg_ls: A list of DNA targets
           
   o Output: A list of calculated combinatorial effects.
'''
def CombineGM(m2_dic,sg_ls,tg_ls):
    cm_ls = []
    for i in range(len(sg_ls)):
        s1 = sg_ls[i][0:20].upper()
        s2 = tg_ls[i][0:20].upper()
        pos_ls = sorted(GetMutType(s1,s2)[0])
        
        ## Combinatorial effect at certain position combination.
        di_ls = list(combinations(pos_ls,2))
        c_ls = [m2_dic[str(di[0])+'&'+str(di[1])] for di in di_ls] 
        
        if len(pos_ls) > 1:
            m = gmean(c_ls)**(len(pos_ls)-1) ## Geometirc mean of all possible combinations
        else:
            m = 1
        cm_ls.append(m)
    return cm_ls


'''This function is predict off-target MOFF score for given gRNA-target pairs

   o input:1). m2_dic: Python dic contains CE of all the possible position combinaitons (20*19)
           2). m1_dic: Python dic contains MDE of all the possible nucleotide mismatches (12) 
               at all possible positions (20)  
           3). df: A panda dataframe with one column of sgRNA and another column of DNA targets
           
   o Output: A panda dataframe with off-target predictions using different models (factors) 
'''
def MOFF_score(m1_dic,m2_dic,df):
    from keras import models
    from keras import layers
    from keras import optimizers
    
    sg_ls = list(df['crRNA']) # Get list of input sgRNAs
    tg_ls = list(df['DNA']) # Get list of input DNA targets
    
    np.random.seed(24) # for reproducibility
    model = models.load_model('./StaticFiles/GOP_model_3.h5')
    pred_test = list(model.predict(OneHotEndocing([s.upper()[0:20] for s in sg_ls])))
    df['GOP'] = [g[0] for g in pred_test]
    
    
    df['MDE'] = Multiply(m1_dic,sg_ls,tg_ls)
    df['CE'] = CombineGM(m2_dic,sg_ls,tg_ls)
    df['MMs'] = MisNum(sg_ls,tg_ls)
    df['MisType'] = MisType(sg_ls,tg_ls)
    df['GMT'] = df['GOP']**df['MMs']
    df['MOFF'] = df['MDE']*df['CE']*df['GMT']
    return df


'''This function is predict off-target MOFF score for given gRNA-target pairs

   o input:1). m2_dic: Python dic contains CE of all the possible position combinaitons (20*19)
           2). m1_dic: Python dic contains MDE of all the possible nucleotide mismatches (12) 
               at all possible positions (20)  
           3). df: A panda dataframe with one column of sgRNA and another column of DNA targets
           
   o Output: A panda dataframe with aggregated off-target scores for each sgRNAs using different models (factors) 
'''
def MOFF_aggregate(m1_dic,m2_dic,df):
    
    sg_ls = []; gmt_ls = []; mde_ls = []; moff_ls = []
    for sg in set(df['crRNA']):  ## Go through all the sgRNAs 
        df_sg = df[df['crRNA']==sg] ## Get all the off-targets for certain sgRNA
        df_score = MOFF_score(m1_dic,m2_dic,df_sg)
        
        sg_ls.append(sg)
        gmt_ls.append(df_score['GOP'].mean())
        ##### Sum up the scores for single gRNA-target pairs ####
        
        if list(df_score['MM']).count(0) >=1:
            mde_ls.append(np.log(df_score['MDE'].sum()-1))
            moff_ls.append(np.log(df_score['MOFF'].sum()-1))
        else:
            mde_ls.append(np.log(df_score['MDE'].sum()))
            moff_ls.append(np.log(df_score['MOFF'].sum()))
    
    df_out = pd.DataFrame({'sgRNA':sg_ls,'GMT':gmt_ls,'MDE.sum':mde_ls,'MOFF.sum':moff_ls})
    return df_out


'''This function is to get reverse complement of a input DNA sequence'''
def ReverseComplement(s):
    x = ''
    for i in range(len(s)):
        if s[i] == 'A': x+='T'
        elif s[i] == 'T': x += 'A'
        elif s[i] == 'G': x += 'C'
        elif s[i] == 'C': x += 'G'
    return  x[::-1] # Reverse 


'''This function is to search all possible sgRNAs targeting the allele

   o input: 1). s1: The local DNA sequence of allele you want to knockouts, usually it is the mutant allele.
            2). s2: The local DNA sequence of allele you want to keep, usually it is the wildtype allele.
            
            Note: the s1 and s2 should be of same length, and at least one 20bp+(NGG)PAM hit should be in s1.
           
   o Output: A list of sgRNA pairs targeting mutant and allele DNA sequences.
'''
def SearchPAM(s1,s2):
    ix = 0
    for i in range(len(s1)):
        if s1[i]!=s2[i]:
            ix = i       ## Get the index of the mutation
    
    pam_ix = [i for i in range(0,len(s1)-2) if s1[i:i+2]=='GG'] ##Searching for all the GG
    pair_ls = []
    for pi in pam_ix:
        if pi>20 and -1 <= pi - ix < 20: ## Make sure we have 20bp before NGG 
            pair_ls.append((s1[pi-21:pi+2],s2[pi-21:pi+2]))
            
    return pair_ls


'''This function is design all possible sgRNAs for allele-specific knockouts by introducing second mismatch.

   o input: 1). sg_ko: sgRNA which is perfect matched to the allele you want to knockout.
            2). sg_mut: sgRNA with one mismatch to the allele you want to keep.
           
   o Output: A dataframe recording all the possible two-mismatches introduced for specific knockouts
'''
def CombinationDesign(sg_ko,sg_mut):
    pos = [i for i in range(len(sg_ko)) if sg_ko[i]!=sg_mut[i]][0] ##Index where mismatch happen
    sg_ls = [sg_ko] ## Add perfect sgRNA first
    for i in range(20):  ## Add all possible 2-mismatch combinations
        sg_com = list(sg_ko)
        if i+1 != pos:
            nt_old = sg_ko[i]
            for nt in ['A','T','C','G']:
                sg_com = list(sg_ko)
                if nt != nt_old:
                    sg_com[i] = nt
                    sg_ls.append(''.join(sg_com))
    
    mut1_ls = []; mut2_ls = []
    df = pd.DataFrame({'sgRNA':sg_ls,'DNA_KO':[sg_ko]*len(sg_ls),'DNA_NA':[sg_mut]*len(sg_ls)})
    return df


'''This function is predict off-target MOFF score for given gRNA-target pairs

   o input:1). m2_dic: Python dic contains CE of all the possible position combinaitons (20*19)
           2). m1_dic: Python dic contains MDE of all the possible nucleotide mismatches (12) 
               at all possible positions (20)  
           3). s1: The local DNA sequence of allele you want to knockouts, usually it is the mutant allele.
           4). s2: The local DNA sequence of allele you want to keep, usually it is the wildtype allele. 
           
   o Output: A panda dataframe with all the possible sgRNAs and MOFF score for mutant and wildtype alleles. 
'''
def MOFF_Allele(m1_dic,m2_dic,s1,s2):
    from keras import models
    from keras import layers
    from keras import optimizers
    
    rs1 = ReverseComplement(s1); rs2 = ReverseComplement(s2)
    logging.info('Searching PAM in the given DNA sequence ...')
    pair_ls = SearchPAM(s1,s2) + SearchPAM(rs1,rs2)
    logging.info('Find %d available sgRNAs'%len(pair_ls))
    df = pd.DataFrame()
    for pair in pair_ls:
        sg_ko = pair[0]; sg_mut = pair[1]
        df_p = CombinationDesign(sg_ko,sg_mut)
        df = pd.concat([df,df_p])
    
    sg_ls = list(df['sgRNA']); d1_ls = list(df['DNA_KO']); d2_ls = list(df['DNA_NA'])
    
    
    np.random.seed(24) # for reproducibility
    model = models.load_model('./StaticFiles/GOP_model_3.h5')
    pred_test = list(model.predict(OneHotEndocing([s.upper()[0:20] for s in sg_ls])))
    df['GMT'] = [g[0] for g in pred_test]
    
    df['MDE_KO'] = Multiply(m1_dic,sg_ls,d1_ls)
    df['CE_KO'] = CombineGM(m2_dic,sg_ls,d1_ls)
    df['MMs_KO'] = MisNum(sg_ls,d1_ls)
    df['MisType_KO'] = MisType(sg_ls,d1_ls)
    df['GMT_KO'] = df['GMT']**df['MMs_KO']
    df['MOFF_KO'] = df['MDE_KO']*df['CE_KO']*df['GMT_KO'] ## MOFF score 
    
    df['MDE_NA'] = Multiply(m1_dic,sg_ls,d2_ls)
    df['CE_NA'] = CombineGM(m2_dic,sg_ls,d2_ls)
    df['MMs_NA'] = MisNum(sg_ls,d2_ls)
    df['MisType_NA'] = MisType(sg_ls,d2_ls)
    df['GMT_NA'] = df['GMT']**df['MMs_NA']
    df['MOFF_NA'] = df['MDE_NA']*df['CE_NA']*df['GMT_NA']
    
    return df.loc[:,['sgRNA','DNA_KO','DNA_NA','MisType_KO','MisType_NA','GMT','MOFF_KO','MOFF_NA']]
    

    

