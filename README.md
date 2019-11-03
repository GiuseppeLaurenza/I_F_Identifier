# I_F_Identifier
This repository contains scripts and dataset for a Apt Identification Triage System. It contains the code from:
* Laurenza, Giuseppe, et al. *Malware triage based on static features and public apt reports.* International Conference on Cyber Security Cryptography and Machine Learning (CSMCL2017). Springer, Cham, 2017.
  * This work is implemented through the class *ThresholdRandomForest* (rf)
* Laurenza, Giuseppe, et al. *Malware triage for early identification of Advanced Persistent Threat activities.* arXiv preprint arXiv:1810.07321 (2018).
  * This work is implemented as *One Class* (oc)

##File Details
* *dataset.tar.gz* contains two hdf files containing features of APT-malware and normal malware
* *test_article.py* contains the code to test each implementation and compute some metrics
* *Checking_Result.py* contains the code to compute metrics on the test
* *ThresholdRandomForest.py* contains all the methods to implement the functionalities of the first work
* _select*_ contain data obtained from our tests about best classes, best parameters and best columns to reproduce the published results
