# Import necessary libraries
import matplotlib.pyplot as plt  # Library for plotting
import numpy as np  # Library for numerical operations
from PIL import Image  # Library to handle images
import pandas as pd  # Library to work with data in Excel files
import openpyxl  # Library to handle Excel files

from sklearn.feature_extraction.text import CountVectorizer  # For text vectorization
from sklearn.decomposition import LatentDirichletAllocation  # For LDA modeling
from collections import defaultdict  # For a dictionary with default values
import spacy  # Library for NLP tasks

# Check if the spaCy model is installed
if 'en_core_web_sm' not in spacy.cli.info()['pipelines']:
    # If not installed, download the model
    spacy.cli.download('en_core_web_sm')

# Load the 'en_core_web_sm' model from spaCy
nlp = spacy.load('en_core_web_sm')

# Part 1 - Image Representation

# Store flattened image arrays
imageArrays = []

# Part 1 Step 1 - Read images and resize
# Loop through each image from 1 to 10
for i in range(1, 11):
    # Load the image
    image = Image.open(str(i) + '.PNG')

    # Resize the image to 100x100 pixels
    resizedImage = image.resize((100, 100))

    # Convert the image to a numpy array
    imageArray = np.array(resizedImage)

    # Part 1 Step 2 - Convert to grayscale
    grayscaleImageArray = np.array(resizedImage.convert('L'))

    # Part 1 Step 3 - Flatten and plot histograms
    # Flatten the grayscale image array and store it
    flattenedImageArray = grayscaleImageArray.flatten()
    imageArrays.append(flattenedImageArray)

    # Plot the histogram for the image
    plt.hist(flattenedImageArray, bins=100)
    plt.title("Histogram for Image {}".format(i))
    plt.savefig("hist_" + str(i) + ".png")
    plt.close()

    # Part 1 Step 4 - Equalize histograms
    # Equalize the histogram of the image
    imageHistogram, bins = np.histogram(flattenedImageArray, 256, density=True)
    cdf = imageHistogram.cumsum()
    cdf = 255 * cdf / cdf[-1]
    equalizedImageArray = np.interp(flattenedImageArray, bins[:-1], cdf)

    # Plot the equalized histogram for the image
    plt.hist(equalizedImageArray, bins=100)
    plt.title("Equalized Histogram for Image {}".format(i))
    plt.savefig("hist_eq_" + str(i) + ".png")
    plt.close()

# Export flattened arrays to CSV
import csv

with open('images.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(imageArrays)

# Part 1 Step 5 - Compare histograms (Histogram differences)
print("The equalized histograms show a better balance in intensity, bringing out more details and enhancing contrast in the images")

# Part 2 - Topic Modeling
# Load data from Excel file
dataFrame = pd.read_excel('Assignment2text.xlsx')

# Preprocess text using spacy
processedDocs = []
for review in dataFrame['review']:
    # Tokenize and lemmatize each word in the review if it's not a stopword or punctuation
    doc = nlp(review)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    # Join the tokens back into a single string and append it to the processedDocs list
    processedDocs.append(' '.join(tokens))

# Part 2 Step 1 - Vectorize documents
documentVectorizer = CountVectorizer(min_df=5, ngram_range=(1, 2))
# Convert the processed text into a matrix of token counts
vectorizedDocuments = documentVectorizer.fit_transform(processedDocs)

# Part 2 Step 2 - Apply LDA
ldaModel = LatentDirichletAllocation(n_components=6)
# Fit the LDA model to the vectorized documents
ldaModel.fit(vectorizedDocuments)

# Part 2 Step 3 - Get topic distributions
# Transform the first 10 and 501-510 documents to their topic distributions using LDA
restaurantTopics = ldaModel.transform(documentVectorizer.transform(processedDocs[:10]))
movieTopics = ldaModel.transform(documentVectorizer.transform(processedDocs[500:510]))

# Part 2 Step 4 - Get top terms for each topic
topicTerms = defaultdict(list)
featureNames = documentVectorizer.get_feature_names_out()  # Get feature names

# For each topic, get the top 5 terms based on their weights in the LDA model
for i, topic in enumerate(ldaModel.components_):
    topIndices = topic.argsort()[-5:]
    topTerms = [featureNames[index] for index in topIndices]
    topicTerms[i] = topTerms

# Print results
print("Restaurant review 1 focuses on topics:", restaurantTopics[0].argmax())
print("Movie review 501 focuses on topics:", movieTopics[0].argmax())

print("Top terms for each topic:")
for i, terms in topicTerms.items():
    print("Topic {}: {}".format(i, terms))

# Part 2 Step 5 - Describe example reviews
print("Review 1 focuses on food descriptions and service quality.")
print("Review 501 focuses on plot, characters, and genres.")
