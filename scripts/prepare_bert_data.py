import sys
import json
from os.path import join, exists
from transformers import BertTokenizer

def tokenize_no_unk(tokenizer, text):
    split_tokens = []
    for token in tokenizer.basic_tokenizer.tokenize(text, never_split=tokenizer.all_special_tokens):
        wp_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        if wp_tokens == [tokenizer.unk_token]:
            split_tokens.append(token)
        else:
            split_tokens.extend(wp_tokens)
    return split_tokens

def find_sublist(a, b):
    if not b: 
        return -1
    for i in range(len(a)-len(b)+1):
        if a[i:i+len(b)] == b:
            return i
    return -1


if __name__ == '__main__':
    
    if len(sys.argv) < 4:
        print('Usage: python3 prepare_bert_data.py <pretrained_model> <split> <dataset_1> <dataset_2> ... <dataset_n>')
        exit(1)

    model_path = sys.argv[1]
    split = sys.argv[2]
    datasets = sys.argv[3:]
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    for dataset in datasets:
        data = json.load(open('dataset/%s.json' % dataset))
        passage_count = len(data)
        impossible_questions = 0
        for i, PQA in enumerate(data, start=1):

            # Passage
            PID = PQA['DID']
            raw_passage = PQA['DTEXT'].strip()
            passage = tokenizer.tokenize(raw_passage)
            passage_no_unk = tokenize_no_unk(tokenizer, raw_passage)
            
            # Add [SEP] to seperate sents in passage
            sep_ind = [idx for idx, token in enumerate(passage) if token == '。']
            for si, idx in enumerate(sep_ind):
                passage.insert(idx+si+1, tokenizer.sep_token)
                passage_no_unk.insert(idx+si+1, tokenizer.sep_token)
                sep_ind[si] += si + 1

            if passage[-1] != tokenizer.sep_token:
                passage.append(tokenizer.sep_token)
                passage_no_unk.append(tokenizer.sep_token)
                sep_ind.append(len(passage)-1)

            # QA pairs
            QAs = []
            for QA in PQA['QUESTIONS']:
                if 'ANSWER' not in QA:
                    impossible_questions += 1
                    continue
                
                processed_QA = {}
                raw_question = QA['QTEXT'].strip()
                question = tokenizer.tokenize(raw_question)
                question_no_unk = tokenize_no_unk(tokenizer, raw_question)

                raw_answers = [A['ATEXT'].strip() for A in QA['ANSWER']]
                answer_no_unk = tokenize_no_unk(tokenizer, raw_answers[0])

                answer_start = find_sublist(passage_no_unk, answer_no_unk)
                answer_end = answer_start + len(answer_no_unk) - 1 if answer_start >= 0 else -1
                if answer_start < 0:
                    impossible_questions += 1
                    
                if answer_start < 0 or answer_end < 0:
                    SE_idx = -1
                else:
                    SE_idx = -1
                    for cur, token in enumerate(passage_no_unk[answer_end+1:], start=answer_end+1):
                        if token == tokenizer.sep_token:
                            SE_idx = cur
                            break

                if answer_start >= 0 or split != 'train':
                    processed_QA['question'] = question
                    processed_QA['question_no_unk'] = question_no_unk
                    processed_QA['answer'] = raw_answers
                    processed_QA['answer_start'] = answer_start
                    processed_QA['answer_end'] = answer_end
                    processed_QA['SE_idx'] = SE_idx
                    processed_QA['id'] = QA['QID']
                    QAs.append(processed_QA)

            # Save processed data
            with open('data/%s/passage/%s|%s' % (split, dataset, PID), 'w') as f:
                assert passage == ' '.join(passage).split(' ')
                f.write(' '.join(passage))

            with open('data/%s/passage_no_unk/%s|%s' % (split, dataset, PID), 'w') as f:
                assert passage_no_unk == ' '.join(passage_no_unk).split(' ')
                f.write(' '.join(passage_no_unk))

            for QA in QAs:
                question = QA['question']
                question_no_unk = QA['question_no_unk']
                answers = QA['answer']
                answer_start = QA['answer_start']
                answer_end = QA['answer_end']
                SE_idx = QA['SE_idx']
                QID = QA['id']
                with open('data/%s/question/%s|%s|%s' % (split, dataset, PID, QID), 'w') as f:
                    assert question  == ' '.join(question).split(' ')
                    f.write(' '.join(question))
                with open('data/%s/question_no_unk/%s|%s|%s' % (split, dataset, PID, QID), 'w') as f:
                    assert question_no_unk  == ' '.join(question_no_unk).split(' ')
                    f.write(' '.join(question_no_unk))
                with open('data/%s/answer/%s|%s|%s' % (split, dataset, PID, QID), 'w') as f:
                    for answer in answers:
                        f.write('%s\n' % answer)
                with open('data/%s/span/%s|%s|%s' % (split, dataset, PID, QID), 'w') as f:
                    f.write('%d %d' % (answer_start, answer_end))
                with open('data/%s/SE_idx/%s|%s|%s' % (split, dataset, PID, QID), 'w') as f:
                    f.write('%d' % SE_idx)

            print('%s: %d/%d (%.2f%%) \r' % (dataset, i, passage_count, 100*i/passage_count), end='')
        print('\nimpossible_questions: %d' % impossible_questions)
    exit(0)