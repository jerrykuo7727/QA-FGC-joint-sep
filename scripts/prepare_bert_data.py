import re
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

def find_sublist(a, b, order=-1):
    if not b: 
        return -1
    counter = 0
    for i in range(len(a)-len(b)+1):
        if a[i:i+len(b)] == b:
            counter += 1
            if counter > order:
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
        is_FGC = dataset.startswith('FGC')
        impossible_questions = 0
        bad_shint = 0
        for i, PQA in enumerate(data, start=1):

            # Passage
            PID = PQA['DID']
            raw_passage = PQA['DTEXT'].strip().replace('\n', '').replace('\r', '').replace('\t', '')
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
                if QA['AMODE'] != 'Single-Span-Extraction' and \
                   'Single-Span-Extraction' not in QA['AMODE'] or \
                   'ANSWER' not in QA:
                    impossible_questions += 1
                    continue
                
                processed_QA = {}
                raw_question = QA['QTEXT'].strip().replace('\n', '').replace('\r', '').replace('\t', '')
                question = tokenizer.tokenize(raw_question)
                question_no_unk = tokenize_no_unk(tokenizer, raw_question)
                
                # Find answer
                raw_answers = [A['ATEXT'].strip().replace('\n', '').replace('\r', '').replace('\t', '') for A in QA['ANSWER']]
                raw_answer_start = QA['ANSWER'][0]['ATOKEN'][0]['start']
                found_answer_starts = [m.start() for m in re.finditer(raw_answers[0], raw_passage)]
                answer_order, best_dist = -1, 10000
                for order, found_start in enumerate(found_answer_starts):
                    dist = abs(found_start - raw_answer_start)
                    if dist < best_dist:
                        best_dist = dist
                        answer_order = order

                answer_no_unk = tokenize_no_unk(tokenizer, raw_answers[0])
                answer_start = find_sublist(passage_no_unk, answer_no_unk, order=answer_order)
                answer_end = answer_start + len(answer_no_unk) - 1 if answer_start >= 0 else -1
                if answer_start < 0:
                    impossible_questions += 1
                
                # Find SEP position of sentence with answer
                if answer_start < 0 or answer_end < 0:
                    SE_idx = 9999
                else:
                    SE_idx = 9999
                    for cur, token in enumerate(passage_no_unk[answer_end+1:], start=answer_end+1):
                        if token == tokenizer.sep_token:
                            SE_idx = cur
                            break
                    assert passage_no_unk[SE_idx] == tokenizer.sep_token

                # Build SEP mask
                sep_mask = []
                for token in passage_no_unk:
                    if token == tokenizer.sep_token:
                        sep_mask.append(1)
                    else:
                        sep_mask.append(0)
                assert len(sep_mask) == len(passage_no_unk)

                # Build SHINT mask
                shint_mask = [0 for n in sep_mask]
                if SE_idx != 9999:
                    assert sep_mask[SE_idx] == 1
                    shint_mask[SE_idx] = 1
                
                if is_FGC:
                    shint_ind = QA['SHINT']
                    raw_shints = [PQA['SENTS'][shint_idx]['text'].strip().replace('\n', '').replace('\r', '').replace('\t', '') for shint_idx in shint_ind]
                else:
                    raw_shints = list(set([SHINT['text'].strip().replace('\n', '').replace('\r', '').replace('\t', '') for SHINT in QA['SHINT']]))

                shints = [tokenize_no_unk(tokenizer, shint) for shint in raw_shints]
                shint_starts = [find_sublist(passage_no_unk, shint) for shint in shints]
                for shint_start in shint_starts:
                    if shint_start < 0:
                        bad_shint += 1
                        break
                    for n_idx, n in enumerate(sep_mask[shint_start:], start=shint_start):
                        if n == 1:
                            assert passage_no_unk[n_idx] == tokenizer.sep_token, passage_no_unk
                            shint_mask[n_idx] = 1
                            break

                if answer_start >= 0 or split != 'train':
                    processed_QA['question'] = question
                    processed_QA['question_no_unk'] = question_no_unk
                    processed_QA['answer'] = raw_answers
                    processed_QA['answer_start'] = answer_start
                    processed_QA['answer_end'] = answer_end
                    processed_QA['SE_idx'] = SE_idx
                    processed_QA['sep_mask'] = sep_mask
                    processed_QA['shint_mask'] = shint_mask
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
                sep_mask = QA['sep_mask']
                shint_mask = QA['shint_mask']
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
                with open('data/%s/sep_mask/%s|%s|%s' % (split, dataset, PID, QID), 'w') as f:
                    f.write(' '.join([str(n) for n in sep_mask]))
                with open('data/%s/shint_mask/%s|%s|%s' % (split, dataset, PID, QID), 'w') as f:
                    f.write(' '.join([str(n) for n in shint_mask]))

            print('%s: %d/%d (%.2f%%) \r' % (dataset, i, passage_count, 100*i/passage_count), end='')
        print('\nimpossible_questions: %d' % impossible_questions)
        print('bad_shint: %d' % bad_shint)
    exit(0)
