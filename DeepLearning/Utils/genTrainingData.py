__author__ = 'neuralconcept'

# ****** Read the two training sets and the test set
#
import pandas as pd
import os
from DeepLearning.Utils import get_dir, CACHE_DIR

#join aLL FILES

print os.path.dirname(__file__)
fout=open(os.path.join(CACHE_DIR, "out_full.csv"),"w")
# first file:
for line in open(os.path.join(CACHE_DIR, "temp0.csv")):
    fout.write(line)
# now the rest:
for num in range(1,38):
    f = open(os.path.join(CACHE_DIR,"temp"+str(num)+".csv"))
    f.next() # skip the header
    for line in f:
         fout.write(line)
    f.close() # not really needed
fout.close()




train = pd.read_csv(os.path.join(os.path.dirname(__file__), CACHE_DIR, 'out_full.csv'), header=0, delimiter="\t",quoting=2)

#print train['Person'][0]
lines=[]
PERSON_FILENAME = "person.snt.aligned"
DOCTOR_FILENAME = "doctor.snt.aligned"

TRAIN_SUFFIX = "_train"
DEV_SUFFIX = "_dev"

PERSON_PATH = os.path.join(CACHE_DIR, PERSON_FILENAME)
DOCTOR_PATH = os.path.join(CACHE_DIR, DOCTOR_FILENAME)

PERSON_TRAIN_PATH = PERSON_PATH + TRAIN_SUFFIX
PERSON_DEV_PATH = PERSON_PATH + DEV_SUFFIX
DOCTOR_TRAIN_PATH = DOCTOR_PATH + TRAIN_SUFFIX
DOCTOR_DEV_PATH = DOCTOR_PATH + DEV_SUFFIX

person_file = open(os.path.join(CACHE_DIR, PERSON_FILENAME), 'w')
doctor_file = open(os.path.join(CACHE_DIR, DOCTOR_FILENAME), 'w')
for x in range(len(train)):
    if train['Person'][x]=='VOICEinOFF':
        newline = str(train['Person'][x])+ " " + str(train['Text'][x]) + " ENDTEXT"
        lines.append(newline)

        #print newline
    if (x>0):
        if train['Person'][x]=='DOCTOR' and train['Person'][x-1]!='VOICEinOFF':
            newline = train['Person'][x-1]+ "PERSON, " + train['Text'][x-1] + " ENDQT" + " " + train['Text'][x] + " ENDTEXT"
            lines.append(newline)
            person_file.write(train['Text'][x-1].strip())
            person_file.write('\n')
            doctor_file.write(train['Text'][x].strip())
            doctor_file.write('\n')
           # print newline

text_file = open("trainingDrWho.csv", "w")
data = pd.DataFrame(lines, columns=['data'])
text_file.write(data.to_csv( sep='\t', quoting =2))
text_file.close()
person_file.close()
doctor_file.close()
