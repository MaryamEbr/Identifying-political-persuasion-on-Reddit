#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import numpy as np
import argparse
import json
import string
import csv
import os

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
FUTURE_TENSE_VERBS = {"'ll", "will", "gonna"}

SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}



def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    feats_i = np.zeros(173)

    bristol_aoa_list = []
    bristol_img_list = []
    bristol_fam_list = []
    warringer_v_list = []
    warringer_a_list = []
    warringer_d_list = []
    word_list = []

    # TODO: Extract features that rely on capitalization.

    going_to_be_flag = 0
    for sent in comment['body'].split('\n'):
        for tok in sent.split(' '):

            if ('/' in tok):

                # for words containing '/'
                if (len(tok.split('/')) > 2 ):
                    word = tok[:tok.rfind('/')]
                    tag = tok.split('/')[-1]
                else:
                    word = tok.split('/')[0]
                    tag = tok.split('/')[1]

                word_list.append(word)
                # feature 1, number of tokens in uppercase (>= letters long)
                #Feature 1: Only consider tokens that are fully alphabetic,
                if (len(word) >= 3 and word.isupper()):
                    feats_i[0] += 1

                # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
                # TODO: Extract features that do not rely on capitalization.
                # lowercase the word
                word = word.lower()

                # feature 2, Number of first-person pronoun
                if (word in FIRST_PERSON_PRONOUNS):
                    feats_i[1] += 1

                # feature 3, Number of second-person pronouns
                if (word in SECOND_PERSON_PRONOUNS):
                    feats_i[2] += 1

                # feature 4,  Number of third-person pronouns
                if (word in THIRD_PERSON_PRONOUNS):
                    feats_i[3] += 1

                # feature 5, Number of coordinating conjunctions
                if (tag == 'CC'):
                    feats_i[4] += 1

                # feature 6, Number of past-tense verbs
                if (tag == 'VBD'):
                    feats_i[5] += 1

                # feature 7,  Number of future-tense verbs
                # 'll, will, gonna, going to VB
                if (word in FUTURE_TENSE_VERBS):
                    feats_i[6] += 1

                if (word == 'going' or word == 'go'):
                    going_to_be_flag = 0
                    going_to_be_flag += 1
                if (word == 'to' and going_to_be_flag == 1):
                    going_to_be_flag += 1
                if (tag == 'VB' and going_to_be_flag == 2):
                    going_to_be_flag = 0
                    feats_i[6] += 1

                # feture 8, Number of commas
                # count only those that are tokenized by spacy to be a comma! (Piazza)
                if (',' in tag):
                    feats_i[7] += 1

                # feature 9, Number of multi-character puctuation tokens
                # Use string.punctuation to get the set of tokens considered as punctuation,
                # and then determine if it is multi-character by checking the length of the lemma. (Piazza)
                punc_list = string.punctuation
                if (len(word) > 1 and all([ch in string.punctuation for ch in word])):
                    feats_i[8] += 1

                # feature 10, Number of common nouns
                # common nouns: NN, NNS
                if (tag == 'NN' or tag == 'NNS'):
                    feats_i[9] += 1

                # feature 11: Number of proper nouns
                # proper nouns: NNP, NNPS
                if (tag == 'NNP' or tag == 'NNPS'):
                    feats_i[10] += 1

                # feature 12: Number of adverbs
                # adverbs: RB, RBR, RBS
                if (tag == 'RB' or tag == 'RBR' or tag == 'RBS'):
                    feats_i[11] += 1

                # feature 13: Number of wh- words
                # wh- words: WDT, WP, WP$, WRB
                if (tag == 'WDT' or tag == 'WP' or tag == 'WP$' or tag == 'WRB'):
                    feats_i[12] += 1

                # feature 14: Number of slang acronyms
                if (word in SLANG):
                    feats_i[13] += 1

                # features 18_29 lists
                if (word in bristol_dict.keys()):
                    bristol_aoa_list.append(bristol_dict[word][0])
                    bristol_img_list.append(bristol_dict[word][1])
                    bristol_fam_list.append(bristol_dict[word][2])

                if (word in warriner_dict.keys()):
                    warringer_v_list.append(warriner_dict[word][0])
                    warringer_a_list.append(warriner_dict[word][1])
                    warringer_d_list.append(warriner_dict[word][2])

    # feature 15: Average length of sentences, in token
    # a list of sentence lengths (in tokens) in the comment
    sentence_length_list = [len([tok for tok in sentence.split(' ') if len(tok) > 0]) for sentence in
                            comment['body'].split('\n')[:-1]]

    # there are empty comments. it raise a warning to mean an empty list and nan as a result.
    # this if is to prevent the nan. this feature for empty comments stays 0.
    if (len(sentence_length_list) != 0):
        feats_i[14] = np.mean(sentence_length_list)

    # feature 16: Average length of tokens, excluding punctuation only tokens, in characters
    # a list of tokens in the comment, punctuations excluded
    token_list_punc_exclude = [tok for tok in word_list if (
            len(tok) > 0 and tok != '\n' and all([ch in string.punctuation for ch in tok]) == False)]

    # there are empty comments (or empty after punc removal). it raise a warning to mean an empty list and nan as a result.
    # this if is to prevent the nan. this feature for empty comments stays 0.
    if (len(token_list_punc_exclude) != 0):
        feats_i[15] = np.mean([len(tok) for tok in token_list_punc_exclude])

    # feature 17: Number of sentences
    # number of \n s in the comment
    feats_i[16] = comment['body'].count('\n')

    # features 18-29
    if (len(bristol_aoa_list) != 0):
        # feature 18: Average of AoA from bristol
        feats_i[17] = np.mean(bristol_aoa_list)
        # feature 19: Average of IMG from bristol
        feats_i[18] = np.mean(bristol_img_list)
        # feature 20: Average of FAM from bristol
        feats_i[19] = np.mean(bristol_fam_list)
        # feature 21: Standard deviation of AoA from bristol
        feats_i[20] = np.std(bristol_aoa_list)
        # feature 22: Standard deviation of IMG from bristol
        feats_i[21] = np.std(bristol_img_list)
        # feature 23: Standard deviation of FAM from bristol
        feats_i[22] = np.std(bristol_fam_list)

    if (len(warringer_v_list) != 0):
        # feature 24: Average of V.Mean.Sum from warringer
        feats_i[23] = np.mean(warringer_v_list)
        # feature 25: Average of A.Mean.Sum from warringer
        feats_i[24] = np.mean(warringer_a_list)
        # feature 26: Average of D.Mean.Sum from warringer
        feats_i[25] = np.mean(warringer_d_list)
        # feature 27: Standard deviation of V.Mean.Sum from warringer
        feats_i[26] = np.std(warringer_v_list)
        # feature 28: Standard deviation of A.Mean.Sum from warringer
        feats_i[27] = np.std(warringer_a_list)
        # feature 29: Standard deviation of D.Mean.Sum from warringer
        feats_i[28] = np.std(warringer_d_list)

    return feats_i


def extract2(feats, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feat: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feat : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''

    if (comment_class == 'Right'):
        feats[29:173] = right_feats[right_ids.index(comment_id)]

    if (comment_class == 'Left'):
        feats[29:173] = left_feats[left_ids.index(comment_id)]

    if (comment_class == 'Center'):
        feats[29:173] = center_feats[center_ids.index(comment_id)]

    if (comment_class == 'Alt'):
        feats[29:173] = alt_feats[alt_ids.index(comment_id)]

    return feats
def main(args):
    # Declare necessary global variables here.
    # global feats
    global warriner_dict
    global bristol_dict
    global right_ids
    global right_feats
    global alt_ids
    global alt_feats
    global center_ids
    global center_feats
    global left_ids
    global left_feats

    # Load data
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173 + 1))

    # load wordlist ???? double check the location
    parent_of_arg_a1dir = os.path.abspath(os.path.join(args.a1_dir, os.pardir))
    reader_bristol = csv.reader(open(parent_of_arg_a1dir+'/Wordlists/BristolNorms+GilhoolyLogie.csv'), delimiter=',')
    reader_warriner = csv.reader(open(parent_of_arg_a1dir+'/Wordlists/Ratings_Warriner_et_al.csv'), delimiter=',')

    bristol_dict = {}
    for row in reader_bristol:
        # skip the header and last empty lines
        if (row[1] != '' and row[1] != 'WORD'):
            bristol_dict[row[1]] = [int(row[3]), int(row[4]), int(row[5])]

    warriner_dict = {}
    for row in reader_warriner:
        # skip the header
        if (row[1] != '' and row[1] != 'Word'):
            warriner_dict[row[1]] = [float(row[2]), float(row[5]), float(row[8])]

    # load LIWC files ??? double check the location
    right_file = open(args.a1_dir+'/feats/Right_IDs.txt', "r")
    right_ids = right_file.read().split("\n")
    right_feats = np.load(args.a1_dir+'/feats/Right_feats.dat.npy')

    alt_file = open(args.a1_dir+'/feats/Alt_IDs.txt', "r")
    alt_ids = alt_file.read().split("\n")
    alt_feats = np.load(args.a1_dir+'/feats/Alt_feats.dat.npy')

    center_file = open(args.a1_dir+'/feats/Center_IDs.txt', "r")
    center_ids = center_file.read().split("\n")
    center_feats = np.load(args.a1_dir+'/feats/Center_feats.dat.npy')

    left_file = open(args.a1_dir+'/feats/Left_IDs.txt', "r")
    left_ids = left_file.read().split("\n")
    left_feats = np.load(args.a1_dir+'/feats/Left_feats.dat.npy')

    cat_map = {'Left': 0, 'Center': 1, 'Right': 2, 'Alt': 3}
    for i, comment in enumerate(data):
        # TODO: Call extract1 for each datatpoint to find the first 29 features.
        # Add these to feats.
        feats_i = extract1(comment)
        feats[i][:-1] = feats_i


        # TODO: Call extract2 for each feature vector to copy LIWC features (features 30-173)
        # into feats. (Note that these rely on each data point's class,
        # which is why we can't add them in extract1).
        feats_i = extract2(feats[i][:-1], comment['cat'], comment['id'])
        feats[i][:-1] = feats_i

        #### add the last col
        feats[i][173] = cat_map[comment['cat']]


    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir",
                        help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.",
                        default="/u/cs401/A1/")
    args = parser.parse_args()

    main(args)
