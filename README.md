# A Robust Model for Domain Recognition of Acoustic Communication using Bi-directional LSTM and Deep Neural Network

## Components:
  <ul>
  <li>A ipynb notebook containing the code for training proposed model.</li>
  <li>utils.py a custom made generic python library used for pre-processing the text.</li>
  <li>A folder dataset containing the 8 different text files each containing some sample sentences for each respective domain.</li>
  </ul>

## Steps for running the notebook:
<ol>
    <li>Change Global variables depending upon the training phase and testing phase.</li>
    <li>Change gloabal variable CREATE_OBJECTS to true in case the objects are not created.</li>
    <li>Once the objects are created copy those objects in the same folder where this notebook is present.</li>
    
</ol>
### Before calling the utils for the first time only follow below steps:

<ol>
    <li>Download the glove evectors from http://nlp.stanford.edu/data/glove.840B.300d.zip</li>
    <li>Unzip the downloaded file</li>
    <li>Copy the path of unziped text file.</li>
    <li>Paste the path at line no. 59 of utils.py</li>
</ol>

