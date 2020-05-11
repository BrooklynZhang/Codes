import csv
import json
from google.colab import files
print("Upload the Json file")
uploaded = files.upload()
labels = [["anger"],["anticipation"],["disgust"],["fear"],["joy"],["love"],["optimism"],["pessimism"],["sadness"],["surprise"],["trust"],["neutral"]]

def generate_csv(file, csvfile, labelfile, validfile):
  data= open(file,"r")
  out = open(csvfile, "w", encoding="utf-8", newline="")
  valid = open(validfile, "w", encoding="utf-8", newline="")
  csv.writer(out).writerow(["id", "text", "anger", "anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust","neutral"])
  csv.writer(valid).writerow(["id", "text", "anger", "anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust","neutral"])
  data = json.load(data)
  id = 0
  for i,v in data.items():
    vector = [0]* 12
    for em, val in enumerate(v['emotion'].values()):
      vector[em] = 1 if val else 0
    csv.writer(out).writerow([id,v['body']]+vector)
    id += 1
    if id <= 100:
      csv.writer(valid).writerow([id,v['body']]+vector)
    
  
  label = open(labelfile, "w", encoding="utf-8", newline="")
  for l in labels:
    csv.writer(label).writerow(l)

  out.close()
  valid.close()
  label.close()

if __name__ == "__main__":
	generate_csv("/content/nlp_train.json", "train.csv", "labels.csv", "valid.csv")