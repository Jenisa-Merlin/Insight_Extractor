import subprocess

# Install necessary packages
def install_packages():
    try:
        subprocess.check_call(['pip', 'install', 'beautifulsoup4', 'pandas', 'nltk', 'openpyxl'])
    except Exception as e:
        print(f"Error installing packages: {str(e)}")

# Check if necessary packages are installed, if not, install them
def check_installation():
    try:
        import bs4
        import pandas
        import nltk
        import openpyxl
    except ImportError:
        print("Some packages are missing. Installing now...")
        install_packages()

# Call the function to check and install necessary packages
check_installation()

# import necessary libraries
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import cmudict, stopwords
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('cmudict')

# function used to extract data from the website
def extractArticleText(url):
  try:
    # response for the url using html parser
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # extract title
    title = soup.find('title').get_text().strip()
    # extract main text content
    articleText = ""
    mainContent = soup.find('div', class_='article-content')
    if mainContent:
      for paragraph in mainContent.find_all('p'):
        articleText += paragraph.get_text() + "\n"
    else:
      for paragraph in soup.find_all('p'):
        articleText += paragraph.get_text() + "\n"
    return title, articleText
  except Exception as e:
    print(f"Error extracting text from {url}: {str(e)}")
    return None, None

# Sentiment analysis
def calculateSentiment(text, positiveWords, negativeWords):
  positiveScore = sum(1 for word in text if word in positiveWords)
  negativeScore = (-1) * sum(-1 for word in text if word in negativeWords)
  polarityScore = (positiveScore - negativeScore) / ((positiveScore + negativeScore) + 0.000001)
  subjectivityScore = (positiveScore + negativeScore) / (len(text) + 0.000001)
  return positiveScore, negativeScore, polarityScore, subjectivityScore

# Load CMU Pronouncing Dictionary for syllable counting
cmuDict = cmudict.dict()
stopWords = set(stopwords.words('english'))

# function for counting syllables
def countSyllables(word):
  if word.lower() in cmuDict:
    return max([len(list(y for y in x if y[-1].isdigit())) for x in cmuDict[word.lower()]])
  else:
    return len(word) // 2

# readability analysis
def calculateReadability(text):
  words = word_tokenize(text)
  sentences = sent_tokenize(text)
  wordCount = len(words)
  sentenceCount = len(sentences)
  avgSentenceLength = wordCount / sentenceCount if sentenceCount > 0 else 0
  complexWordCount = sum(1 for word in words if countSyllables(word) > 2)
  complexWordsPercentage = (complexWordCount / wordCount)
  fogIndex = 0.4 * (avgSentenceLength + complexWordsPercentage)
  avgWordsPerSentence = wordCount / sentenceCount if sentenceCount > 0 else 0
  syllablePerWord = sum(countSyllables(word) for word in words) / wordCount if wordCount > 0 else 0
  personalPronouns = sum(1 for word in words if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'])
  avgWordLength = sum(len(word) for word in words) / wordCount if wordCount > 0 else 0
  return avgSentenceLength, complexWordsPercentage, fogIndex, avgWordsPerSentence, complexWordCount, wordCount, syllablePerWord, personalPronouns, avgWordLength

# main function
def main():
  inputFile = "Input.xlsx"
  extractionFolder = "Extracted_Articles/"

  # create folder if not exists
  if not os.path.exists(extractionFolder):
    os.makedirs(extractionFolder)

  # read URLs from Excel file
  df = pd.read_excel(inputFile)

  for index, row in df.iterrows():
    urlID = row['URL_ID']
    url = row['URL']
    title, articleText = extractArticleText(url)

    if title and articleText:
      with open(extractionFolder + f"{urlID}.txt", 'w', encoding='utf-8') as f:
        f.write(f"Title: {title}\n\n")
        f.write(articleText)
      print(f"Article extracted from {url} and saved as {urlID}.txt")
    else:
      print(f"Failed to extract article from {url}")

  stopWordsFolder = "StopWords"

  # load stop words
  stopWordsList = set()
  for stopFileName in os.listdir(stopWordsFolder):
    try:
      with open(os.path.join(stopWordsFolder, stopFileName), 'r', encoding='utf-8') as stopFile:
        stopWordsList.update(stopFile.read().splitlines())
    except UnicodeDecodeError:
      with open(os.path.join(stopWordsFolder, stopFileName), 'r', encoding='latin-1') as stopFile:
        stopWordsList.update(stopFile.read().splitlines())

  positiveWordsPath = "MasterDictionary\positive-words.txt"
  negativeWordsPath = "MasterDictionary\\negative-words.txt"

  # positive and negative dictionary
  positiveWords = set()
  negativeWords = set()

  # Load positive words from file
  with open(positiveWordsPath, 'r', encoding='utf-8') as file:
    for line in file:
      word = line.strip()
      if word:
        positiveWords.add(word.lower())

  # Load negative words from file
  with open(negativeWordsPath, 'r', encoding='latin-1') as file:
    for line in file:
      word = line.strip()
      if word:
        negativeWords.add(word.lower())

  outputFile = "Output.xlsx"
  # Load input data
  inputDf = pd.read_excel(inputFile)
  outputRows = []

  # iterate over each row in input Dataframe
  for index, row in inputDf.iterrows():
    urlID = row['URL_ID']
    filePath = os.path.join(extractionFolder, f"{urlID}.txt")

    # Check if file exists before attempting to open
    if not os.path.exists(filePath):
      print(f"Text file for URL ID {urlID} does not exist. Skipping.")
      continue
    
    with open(filePath, 'r', encoding='utf-8') as file:
      articleText = file.read()

    cleanedText = re.sub(r'[^\w\s]', '', articleText.lower())
    cleanedTextTokens = word_tokenize(cleanedText)
    cleanedTextTokens = [word for word in cleanedTextTokens if word not in stopWordsList]

    positiveScore, negativeScore, polarityScore, subjectivityScore = calculateSentiment(cleanedTextTokens, positiveWords, negativeWords)
    avgSentenceLength, complexWordsPercentage, fogIndex, avgWordsPerSentence, complexWordCount, wordCount, syllablePerWord, personalPronouns, avgWordLength = calculateReadability(articleText)


    # Append metrics to output list
    outputRow = {**row, **{"POSITIVE SCORE": positiveScore, "NEGATIVE SCORE": negativeScore, "POLARITY SCORE": polarityScore, "SUBJECTIVITY SCORE": subjectivityScore, "AVG SENTENCE LENGTH": avgSentenceLength, "PERCENTAGE OF COMPLEX WORDS": complexWordsPercentage, "FOG INDEX": fogIndex, "AVG NUMBER OF WORDS PER SENTENCE": avgWordsPerSentence, "COMPLEX WORD COUNT": complexWordCount, "WORD COUNT": wordCount, "SYLLABLE PER WORD": syllablePerWord, "PERSONAL PRONOUNS": personalPronouns, "AVG WORD LENGTH": avgWordLength}}
    outputRows.append(outputRow)

  # Convert list of dictionaries to DataFrame
  outputDf = pd.DataFrame(outputRows)
  # Save output to Excel file
  outputDf.to_excel(outputFile, index=False)

if __name__ == "__main__":
  main()