# Chronic kidney disease classification by Decision tree - Binary_Classification
## Objectives
This work aims to perform binary classification of chronic kidney disease using the C4.5 algorithm of scikit-learn. Here I need to estimate the last column from the remained features using the Scikit Learn Python library that implements C4.5 or similar algorithm for hierarchical classification.

The C4.5 algorithm allows the generation of a decision tree based on the normalized mutual information or what is also called normalized information gain as a classification criteria to assign the convenient attribute at each node of the tree and to classify patients. The generated decision tree is easily read and understood.

Besides, the Scikit Learn implementation requires numerical data only so all categorical features must be mapped into numbers.
•	Yes to 1 and no to 0, Normal to 1 and abnormalto 0, etc.
## Dataset description:
The first 29 rows of the provided dataset file for this lab contain the description of the features and they must be skipped. The total dataset contains 400 instances classified into two classes as follow: 250 corresponding to the CKD and 150 not CKD.

From the provided description, it is noticed the existence of 24 features plus the last field (column 25) that corresponds to the class which specifies if the disease is present (ckd) or not (notckd). In the 24 features, it exists 11 numerical features and the rest are categorical or nominal features. 

Before the analyzing phase, the dataset must be prepared and cleaned due to the existence of some errors and missing fields. This task is usually needed and sometimes requires a long period. Moreover, the dataset contains some rows with an extra separator field “,”, so 26 columns were read instead of 25. The missing fields were identified by“?” and at the end the categorical features must be transformed into numerical but before that there are “hidden” typing error corresponds to typing “ yes” and not “yes” must be deleted.
Two options are exist for cleaning the data: manually by editing the original CSV file but it is preferable if the file is short while in our case it is better to exploit arguments of pandas.

In order to manage the NaN values the following two approaches were applied:
•	Removing the rows containing NaN values using the methods dropna of Pandas. 
•	Treating NaN values as another possible random variable but must be substituted with a number not already presented in the dataset. In this report -3 was chosen as random value.
## The Chronic kidney disease (CKD) (it was mentioned in my clustering repository also ):
A disease that affects the Kidney’s functionality. The kidney may loses its function for a period of months or years. This kind of disease has many causes where the major one is diabetes, high blood pressure, glomerulonephritis and polycystic kidney disease. Diagnosis is generally by blood tests to measure the glomerular filtration rate and urine tests to measure albumin.

## Results and discussion
The next Figures were obtained after executing my python script in spyder IDE:

###### Figure 1: Case1 : Removing the rows containing NaN values
![alt text](https://github.com/BaddyMAK/Classification-with-ML-/blob/main/results/Case%201.png)

###### Figure 2: Case2 : Substituting NaN value with -3 (random value)
![alt text](https://github.com/BaddyMAK/Classification-with-ML-/blob/main/results/Case%202.png)


The above classification trees were obtained by graphviz and saved as dot files. Then those dot files were converted into png by executing the next instruction in my command line : 

!dot -Tpng tree.dot -o tree.png 

The Figure 2 demonstrates how substituting the missing values with random number has given a larger and a more complex decision tree while removing the lines containing miss-ing data has given a simpler tree as shown in Figure 1.

This is due to the higher number of rows in case 2 (Figure 2), which comports 400 samples, but in the case 1 (Figure 1), the number of samples was only 157. The most important feature for classification in case 1 (Figure 1) was the “al” (albumin). In case 2 (Figure 2), the added “-3” value had influenced the hierarchy of the tree where the most important features in classification became as follow: the “hemo” is the most important followed by the “sg” (specific gravity) at second rank and “al” (albumin) third rank. 
Moreover, one of the thresholds for the “al” feature has become negative (al <-1.5) means the -3 was combined with one of the classification attributes.

The two figures: 3 and 4 have illustrated the importance of each features in the dataset for both cases “Remove” and “Substitution” confirming the interpretation of the two above de-cision trees (Figure 2 and Figure 1).  In case 1, the most important feature is number 3, which is the “al” means the classification of all patients was based on only one feature while for the case 2 the most important feature is the “hemo” which is the feature number 14 fol-lowed by “sg” ( feature 2) and “al” (feature 3).

###### Figure 3: Features importance in case of substitution
![alt text](https://github.com/BaddyMAK/Classification-with-ML-/blob/main/results/feature%20importance%20substitution.png)

###### Figure 4: Features importance in case of remove
![alt text](https://github.com/BaddyMAK/Classification-with-ML-/blob/main/results/feature%20importance%20remove.png)

