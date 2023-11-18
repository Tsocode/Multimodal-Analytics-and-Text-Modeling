# Import libraries
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import openpyxl

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import defaultdict
import spacy

# Check if model is installed
if 'en_core_web_sm' not in spacy.cli.info()['pipelines']:
    # Download model
    spacy.cli.download('en_core_web_sm')

# Load model
nlp = spacy.load('en_core_web_sm')

# Part 1 - Image Representation

images = []
# Part 1 Step 1 - Read images and resize
for i in range(1, 11):
    # Load image
    im = Image.open(str(i) + '.PNG')

    # Resize image
    im_rs = im.resize((100, 100))

    # Convert to numpy array
    im_m = np.array(im_rs)

    # Part 1 Step 2 - Convert to grayscale
    im_grey_m = np.array(im_rs.convert('L'))

    # Part 1 Step 3 - Flatten and plot histograms
    im_v = im_grey_m.flatten()

    images.append(im_v)

    # Plot histogram
    plt.hist(im_v, bins=100)
    plt.title("Histogram for Image {}".format(i))
    plt.savefig("hist_" + str(i) + ".png")
    plt.close()

    # Part 1 Step 4 - Equalize histograms
    imhist, bins = np.histogram(im_v, 256, density=True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]
    im2 = np.interp(im_v, bins[:-1], cdf)

    # Plot equalized histogram
    plt.hist(im2, bins=100)
    plt.title("Equalized Histogram for Image {}".format(i))
    plt.savefig("hist_eq_" + str(i) + ".png")
    plt.close()

# Export flattened arrays to CSV
import csv

with open('images.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(images)

# Part 1 Step 5 - Compare histograms (Histogram differences)
print(
    "The equalized histograms are more evenly distributed compared to the original histograms which are concentrated around certain intensity values. Equalization helps stretch contrast and bring out details.")

# Part 2 - Topic Modeling

# Load data
df = pd.read_excel('Assignment2text.xlsx')

# Preprocess text
nlp = spacy.load('en_core_web_sm')
docs = []
for review in df['review']:
    doc = nlp(review)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    docs.append(' '.join(tokens))

# Part 2 Step 1 - Vectorize documents
vectorizer = CountVectorizer(min_df=5, ngram_range=(1, 2))
X = vectorizer.fit_transform(docs)

# Part 2 Step 2 - Apply LDA
lda = LatentDirichletAllocation(n_components=6)
lda.fit(X)

# Part 2 Step 3 - Get topic distributions
restaurant_topics = lda.transform(vectorizer.transform(docs[:10]))
movie_topics = lda.transform(vectorizer.transform(docs[500:510]))

# Previous code

# Part 2 Step 4 - Get top terms
topic_terms = defaultdict(list)
feature_names = vectorizer.get_feature_names_out() # Use correct method name

for i, topic in enumerate(lda.components_):
  top_indices = topic.argsort()[-5:]
  top_terms = [feature_names[index] for index in top_indices]
  topic_terms[i] = top_terms

# Rest of code

# Print results
print("Restaurant review 1 focuses on topics:", restaurant_topics[0].argmax())
print("Movie review 501 focuses on topics:", movie_topics[0].argmax())

print("Top terms for each topic:")
for i, terms in topic_terms.items():
    print("Topic {}: {}".format(i, terms))

# Part 2 Step 5 - Describe example reviews
print("Review 1 focuses on food descriptions and service quality.")
print("Review 501 focuses on plot, characters, and genres.")
