# txt-naive-classifier
A naive machine learning classifier for text files, based on titles in training set.

Given a collection of text files in the directory `txt/`, this program first mines the filenames for keywords and uses the files with words in their file names as a training set for machine learning classification.  It vectorizes the word frequencies (using TfIdf), and among the text files whose filenames do not contain words, identifies the closest match in the training set.

The purpose of this script was to organize a heterogeneous collection of thousands of pdf files that I had accumulated over the years.  First, the pdf files were passed through an ocr utility to convert to text (extension `.pdf.txt`).  Then, once converted to text, and the directory containing those text files was used as the input to this script for processing.
