import transformers as ts
import datasets as ds
import pickle

#TODO - improve readability by adding argument parsing

# SET DIRS
# Set directories to wherever the data is stored after preprocessing based on:

raw_data_dir = "/mnt/sdd/niallt/bio-lm/data/tasks/I2B22010NER/" #CHANGEME
save_data_dir = "/mnt/sdd/niallt/bio-lm/data/tasks/i2b2-2010_hf_dataset/" #CHANGEME

allLabels = []

dataDict = {}

def load_ner_dataset(path, subset):

  dataDict[subset] = {
      "tokens": [],
      "ner_tags_str": [],
      "ner_tags": [],
  }

  lines = []

  with open(path, mode="r") as f:
    lines = f.readlines()


  sentences = []
  labels = []

  currentSampleTokens = []
  currentSampleLabels = []

  for line in lines:
      if line.strip() == "":
          sentences.append(currentSampleTokens)
          labels.append(currentSampleLabels)
          currentSampleTokens = []
          currentSampleLabels = []
      else:
          cleanedLine = line.replace("\n", "")
          token, label = cleanedLine.split(
              " ")[0].strip(), cleanedLine.split(" ")[1].strip()
          currentSampleTokens.append(token)
          currentSampleLabels.append(label)
          allLabels.append(label)

  dataDict[subset]["tokens"] = sentences
  dataDict[subset]["ner_tags_str"] = labels

# Set directories to wherever the data is stored after preprocessing based on: https://github.com/facebookresearch/bio-lm/tree/main/preprocessing
load_ner_dataset(f"/mnt/sdd/niallt/bio-lm/data/tasks/I2B22010NER/train.txt.conll", "train")
load_ner_dataset(f"/mnt/sdd/niallt/bio-lm/data/tasks/I2B22010NER/dev.txt.conll", "validation")
load_ner_dataset(f"/mnt/sdd/niallt/bio-lm/data/tasks/I2B22010NER/test.txt.conll", "test")

allLabels = list(set(allLabels))
label_to_index = {label: index for index, label in enumerate(allLabels)}

for key, value in dataDict.items():
  dataDict[key]["ner_tags"] = [[label_to_index[label] for label in str_labels] for str_labels in value["ner_tags_str"]]
  dataDict[key] = ds.Dataset.from_dict(dataDict[key])

dataDict["info"] = ds.Dataset.from_dict({"all_ner_tags": [allLabels]})

dataset = ds.DatasetDict(dataDict)

print(label_to_index)

#print(dataset)

dataset.save_to_disk(f"{save_data_dir}")
