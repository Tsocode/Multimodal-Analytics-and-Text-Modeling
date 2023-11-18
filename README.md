# Multimodal-Analytics-and-Text-Modeling
An integrated project showcasing image processing, histogram equalization, LDA topic modeling, and review analysis using Python's data science libraries

#Multimodal Analytics and Text Modeling
#Overview
This project demonstrates multimodal analytics involving image processing, histogram equalization, and text analysis using Python's data science libraries. It showcases techniques like image resizing, grayscale conversion, histogram visualization, and Latent Dirichlet Allocation (LDA) topic modeling for textual data.

#How to Run
To replicate the results, follow these steps:

Install necessary libraries (e.g., numpy, pandas, scikit-learn, matplotlib, PIL, spacy).
Clone this repository.
Ensure the required datasets (images and text file) are available locally.
Run the provided Python script OmarAssignment2CodePart1.py.
Dependencies
This project requires the following Python libraries:

numpy
pandas
scikit-learn
matplotlib
PIL
spacy
openpyxl
Datasets
Images: Collection of 10 images in PNG format (named 1.PNG to 10.PNG).
Text Data: Text file Assignment2text.xlsx containing 1000 reviews for restaurants and movies.
Code Explanation
Part 1 - Image Representation
Step 1: Read and resize images to 100x100 pixels.
Step 2: Convert images to grayscale.
Step 3: Flatten arrays and plot histograms for intensity value distribution.
Step 4: Perform histogram equalization for normalization.
Step 5: Compare histograms pre- and post-equalization.
Part 2 - Topic Modeling
Step 1: Vectorize text documents for LDA topic modeling.
Step 2: Apply LDA with 6 components for topic extraction.
Step 3: Obtain topic distributions for restaurant and movie reviews.
Step 4: Identify top terms for each topic.
Step 5: Describe example reviews based on identified topics.
Results
Insights gained from image processing and text analysis include:

Enhanced image histograms post-equalization for better intensity distribution.
Identification of key topics within restaurant and movie reviews.
Contributing
Contributions, suggestions, and feedback are welcome. Fork the repository, make changes, and submit pull requests.
