import sys
reload(sys)
sys.setdefaultencoding("utf-8")

from bs4 import BeautifulSoup
import urllib2
import re
import pandas as pd
import cchardet


def convert_encoding(data, new_coding='UTF-8'):
    encoding = cchardet.detect(data)['encoding']

    if new_coding.upper() != encoding.upper():
        data = data.decode(encoding, data).encode(new_coding)

    return data



def process_url(url, data):
    response = urllib2.urlopen(url)
    html = response.read()
    html = convert_encoding(html)
    text = html.replace('/n/n', ' ')
    text = text.rstrip('\r\n')
    text = text.replace("\r\n", " ")
    text = text.replace('<br>', '<br> \r\n')
    text = text.replace('<p><font face="Arial, Helvetica, sans-serif" size="2">',
                        '\r\n <p><font face="Arial, Helvetica, sans-serif" size="2"> ')
    try:
        review_text = BeautifulSoup(text).get_text().encode('windows-1252')
    except:
        review_text = BeautifulSoup(text).get_text()
    review_text = review_text.replace('\n\n', '\n')
    review_text = review_text.replace("\r\n\n", "\n")
    review_text = review_text.replace("\r\r", "\r")
    review_text = review_text.replace("\r\n", "\n")
    lines = review_text.split("\n")
    # Generation
    temporada = temp + 1
    date_emision = ""
    episode = x + 1
    name_episode = ""
    person = ""
    text = ""
    place = ""
    data_row = [temporada, date_emision, episode, name_episode, person, place, text]
    for line in lines:
        newline = line.strip()

        count = 0

        if len(newline) > 0:
            # if (newline.lower().find('episode', 0, 7) >= 0):
            # print "nada interesante"
            if (newline.lower().find('original airdate:') >= 0):
                # print "nada interesante"
                date_emision = newline[newline.find(':') + 2:len(newline)]

            if (newline.find('The Doctor Who Transcripts') >= 0):
                # print 'comienzo'

                items = newline[len('The Doctor Who Transcripts'):len(newline)].split()
                unique = []
                helperset = set()
                for tit in items:
                    if tit not in helperset:
                        unique.append(tit)
                        helperset.add(tit)

                for word in helperset:
                    name_episode = name_episode + word + " "

            if newline[0] == '(':
                # print "contexto"
                person = "VOICEinOFF"
                text = newline[1:len(newline) - 1]
                data_row = [temporada, date_emision, episode, name_episode, person, place, text]
                data.loc[len(data)] = data_row

            if (newline.find('DOCTOR 2:') >= 0) or (len(re.findall(r'([A-Z\d])+:', newline)) > 0):
                # print " actor"
                person = newline[0: newline.find(':')]
                text = newline[newline.find(':') + 2:len(newline)]
                data_row = [temporada, date_emision, episode, name_episode, person, place, text]
                data.loc[len(data)] = data_row

            if newline[0] == '[':
                place = newline[1:len(newline) - 1]
                # print "lugar"


                # \b[A-Z][A-Z0-9]+\b


                # print newline

                #print '\n'

    return data

if __name__ == '__main__':

    columns = ['temporada', 'date', 'episode', 'name_episode', "Person", "Place", "Text"]


    for temp in range(39):
        data = pd.DataFrame(columns=columns)

        for x in range(20):
            print "Processing temp "+ str(temp+1) +" Chapter " + str(x+1)
            try:
                url = 'http://www.chakoteya.net/doctorwho/' + str(temp + 1) + '-' + str(x + 1) + '.htm'
                print url

                data = process_url(url,data)
            except urllib2.HTTPError as e:
                try:
                    url = 'http://www.chakoteya.net/doctorwho/' + str(temp + 1) + '-' + str(x + 1) + '.html'
                    print url
                    data = process_url(url,data)
                except urllib2.HTTPError as e:
                    break



        text_file = open("temp" + str(temp) + ".csv", "w")
        text_file.write(data.to_csv( sep='\t', quoting =2))

        text_file.close()
