# Analyze Arsenic Contamination Levels in Ground Water
![image](https://github.com/diego-lazaro/Project_4/assets/115186079/5afa800f-deb0-45ac-8a70-7ca54cdfab80)



# Synopsis
Analyzed arsenic contamination levels of ground water within the United States. Utilized Machine Learning, specifically logistic regression, to train 3 different models on a  spectrum of contamination. The three spectrums created were, a 1942 Standard in United States of 50㎍/L, a current United States aresenic standard of less than 10㎍/L, and concentrations less than 5㎍/L. Challenged ourselves to create an app that when specific area coordinates are inputted, the model will predict what spectrum the arsenic concentration falls into. 


# Data Processing 
This dataset is intended to represent the potable ground-water resource.
This dataset therefore does not include thermal and saline water (temperature greater than 50℃ or dissolved solids greater than 3000 ㎎/L or specific conductance greater than 4000 µS/㎝).
In addition, this dataset includes only the most recent arsenic analysis available for each well, and only analyses performed by hydride generation or ICP/MS.

# Data Cleaning

![image](https://github.com/diego-lazaro/Project_4/assets/115186079/84f70dc4-41c6-4654-b6e0-b4b4b0150656)

# Arsenic Concentration in the United States

![image](https://github.com/diego-lazaro/Project_4/assets/115186079/6256e679-38df-4a92-900a-88d597b4579e)

# Conclusion
We were able to create a successful app that was capable of predicting the following spectrums with varying accuracies: 
- The inputted coordinates had an arsenic level on the spectrum from 10-50 ㎍/L, the model was 98.06% accurate in its prediction.
- The inputted coordinates had an arsenic level on the spectrum from 5-10 ㎍/L, the model was 87.92% accurate in its prediction.
- The inputted coordinates had an arsenic level on the spectrum from 0-5 ㎍/L, the model was 78.41%  accurate in its prediction.



