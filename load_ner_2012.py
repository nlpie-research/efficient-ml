import transformers as ts
import datasets as ds

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
  
load_ner_dataset("/mnt/sdd/niallt/bio-lm/data/tasks/I2B22012NER/train.txt.conll", "train")
load_ner_dataset("/mnt/sdd/niallt/bio-lm/data/tasks/I2B22012NER/dev.txt.conll", "validation")
load_ner_dataset("/mnt/sdd/niallt/bio-lm/data/tasks/I2B22012NER/test.txt.conll", "test")

allLabels = list(set(allLabels))
label_to_index = {label: index for index, label in enumerate(allLabels)}

for key, value in dataDict.items():
  dataDict[key]["ner_tags"] = [[label_to_index[label] for label in str_labels] for str_labels in value["ner_tags_str"]]
  dataDict[key] = ds.Dataset.from_dict(dataDict[key])

dataDict["info"] = ds.Dataset.from_dict({"all_ner_tags": [allLabels]})

dataset = ds.DatasetDict(dataDict)

print(dataset)

dataset.save_to_disk("/mnt/sdd/niallt/bio-lm/data/tasks/i2b2-2012_hf_dataset/")

loaded_dataset = ds.load_dataset("/mnt/sdd/niallt/bio-lm/data/tasks/i2b2-2012_hf_dataset/")

print(f"loaded_dataset: {loaded_dataset}")


