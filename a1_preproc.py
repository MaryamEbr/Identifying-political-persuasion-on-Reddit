#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz
import html
import sys
import argparse
import os
import json
import re
import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.add_pipe('sentencizer')


def preproc1(comment, steps=range(1, 6)):
    """ This function pre-processes a single comment

    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    """
    modComm = comment
    if 1 in steps:
        # replace newlines with spaces
        modComm = re.sub(r"\n{1,}", " ", modComm)
        modComm = ' '.join(modComm.split())

    if 2 in steps:  # unescape html
        modComm = html.unescape(modComm)

    if 3 in steps:  # remove URLs
        modComm = re.sub(r"\b(http:\/\/|https:\/\/|www\.)\S+", "", modComm)
        ### used a code from internet
        modComm = re.sub(
            re.compile(r"\]?(https?:\/\/|www\.)[a-z0-9\\!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]+", re.IGNORECASE), "",
            modComm)

    if 4 in steps:  # remove duplicate spaces
        modComm = ' '.join(modComm.split())

    if 5 in steps:
        # TODO: get Spacy document for modComm
        # TODO: use Spacy document for modComm to create a string.
        # Make sure to:
        #    * Insert "\n" between sentences.
        #    * Split tokens with spaces.
        #    * Write "/POS" after each token.
        new_modComm = ""
        utt = nlp(modComm)
        for sent in utt.sents:
            for token in sent:
                lemma_temp = ""
                # if the lemma begins with a dash ('-') when the token doesn't, just keep the token
                # I didn't see anything with this condition in max 15000, still just in case..
                if (token.lemma_[0] == '-' and token.text[0] != '-'):
                    lemma_temp = token.text
                # retain the case of the original token. but, if the original token is entirely in uppercase,
                # then so is the lemma; otherwise, keep the lemma in lowercase.
                if (token.text.isupper()):
                    lemma_temp = token.lemma_.upper()
                else:
                    lemma_temp = token.lemma_.lower()

                new_modComm += lemma_temp + '/' + token.tag_ + ' '

            new_modComm = new_modComm[:-1]
            new_modComm += '\n'
    return new_modComm


def main(args):
    ID = int(args.ID[0])
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines
            # sampling starts from Id with circular list indexing
            start_ind = ID % len(data)
            end_ind = start_ind + args.max
            if (end_ind < len(data)):
                data = data[start_ind:end_ind]
            else:
                data = data[start_ind:] + data[:end_ind - len(data)]

            # just checking, delete later ???? *****
            if (len(data) != args.max):
                print("EEEEEEEEror  ", "ars max is bigger than the size of file")

            # TODO: read those lines with something like `j = json.loads(line)`
            data_dict = []
            for line in data:
                data_dict.append(json.loads(line))

            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
            for i in range(len(data_dict)):
                new_body = preproc1(data_dict[i]['body'], steps=[1, 2, 3, 4, 5])
                data_dict[i] = {'id': data_dict[i]['id'], 'body': new_body, 'cat': file}

            # TODO: append the result to 'allOutput'
            allOutput.extend(data_dict)

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir",
                        help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.",
                        default='/u/cs401/A1')

    args = parser.parse_args()

    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    indir = os.path.join(args.a1_dir, 'data')
    main(args)
