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


if __name__ == '__main__':
    
    if len(sys.argv) < 3:
        print('Usage: python3 prepare_bert_data.py <pretrained_model> <dataset_1> <dataset_2> ... <dataset_n>')
    model_path = sys.argv[1]
    datasets = sys.argv[2:]
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    for dataset in datasets:
        print("Prepare data of %s dataset..." % dataset)
        
        # Try to read json files
        datas = {}
        if exists(join('dataset', dataset, '%s_training.json' % dataset)):
            datas['train'] = json.loads(open(join('dataset', dataset, '%s_training.json' % dataset)).read())
        else:
            print("'%s' doesn't exist, skipping..." % (join('dataset', dataset, '%s_training.json' % dataset)))
        if exists(join('dataset', dataset, '%s_dev.json' % dataset)):
            datas['dev'] = json.loads(open(join('dataset', dataset, '%s_dev.json' % dataset)).read())
        else:
            print("'%s' doesn't exist, skipping..." % (join('dataset', dataset, '%s_dev.json' % dataset)))
        if exists(join('dataset', dataset, '%s_test.json' % dataset)):
            datas['test'] = json.loads(open(join('dataset', dataset, '%s_test.json' % dataset)).read())
        else:
            print("'%s' doesn't exist, skipping..." % (join('dataset', dataset, '%s_test.json' % dataset)))

        # Prepare data of each split
        for split in ['train', 'dev', 'test']:
            if split not in datas:
                continue
            bad_context, bad_question = 0, 0
            doc_count = len([paragraph for data in datas[split]['data'] for paragraph in data['paragraphs']])
            parsed_count = 0

            for data in datas[split]['data']:
                for paragraph in data['paragraphs']:

                    # Document
                    raw_context = paragraph['context']
                    cursor = 0
                    context, context_no_unk = [], []
                    answers = []

                    # QA pairs
                    qas = []
                    for qa in sorted(paragraph['qas'], key=lambda x: x['answers'][0]['answer_start']):
                        processed_qa = {}
                        raw_question = qa['question']
                        question = tokenizer.tokenize(raw_question)
                        question_no_unk = tokenize_no_unk(tokenizer, raw_question)
                        assert len(question) == len(question_no_unk)
                        assert tokenizer.unk_token not in question_no_unk

                        raw_answer = qa['answers'][0]['text']
                        raw_answers = [ans['text'] for ans in qa['answers']]
                        raw_answer_start = qa['answers'][0]['answer_start']
                        assert raw_answer in raw_context

                        pre_raw_context = raw_context[cursor:raw_answer_start]
                        pre_context = tokenizer.tokenize(pre_raw_context)
                        pre_context_no_unk = tokenize_no_unk(tokenizer, pre_raw_context)

                        answer = tokenizer.tokenize(raw_answer)
                        answer_no_unk = tokenize_no_unk(tokenizer, raw_answer)
                        assert len(answer) == len(answer_no_unk)
                        assert tokenizer.unk_token not in answer_no_unk
                        if len(answer_no_unk) == 0:
                            bad_question += 1
                            continue

                        context += pre_context
                        context_no_unk += pre_context_no_unk

                        answer_start = len(context)
                        answer_end = answer_start + len(answer) - 1
                        context += answer
                        context_no_unk += answer_no_unk

                        cursor = raw_answer_start + len(raw_answer)        
                        answers.append((answer_no_unk, answer_start, answer_end))

                        processed_qa['question'] = question
                        processed_qa['question_no_unk'] = question_no_unk
                        processed_qa['answer'] = raw_answers
                        processed_qa['answer_start'] = answer_start
                        processed_qa['answer_end'] = answer_end
                        processed_qa['id'] = qa['id']
                        qas.append(processed_qa)

                    # Auto-tests
                    assert len(context) == len(context_no_unk)
                    assert tokenizer.unk_token not in context_no_unk
                    for answer_no_unk, answer_start, answer_end in answers:
                        assert answer_no_unk == context_no_unk[answer_start:answer_end+1]
                    if len(context_no_unk) == 0:
                        bad_context += 1
                        continue

                    # Save processed data
                    with open('data/%s/context/%s|%s' % (split, dataset, paragraph['id']), 'w') as f:
                        assert context == ' '.join(context).split(' ')
                        f.write(' '.join(context))

                    with open('data/%s/context_no_unk/%s|%s' % (split, dataset, paragraph['id']), 'w') as f:
                        assert context_no_unk == ' '.join(context_no_unk).split(' ')
                        f.write(' '.join(context_no_unk))

                    for qa in qas:
                        question = qa['question']
                        question_no_unk = qa['question_no_unk']
                        answers = qa['answer']
                        answer_start = qa['answer_start']
                        answer_end = qa['answer_end']
                        with open('data/%s/question/%s|%s|%s' % (split, dataset, paragraph['id'], qa['id']), 'w') as f:
                            assert question  == ' '.join(question).split(' ')
                            f.write(' '.join(question))
                        with open('data/%s/question_no_unk/%s|%s|%s' % (split, dataset, paragraph['id'], qa['id']), 'w') as f:
                            assert question_no_unk  == ' '.join(question_no_unk).split(' ')
                            f.write(' '.join(question_no_unk))
                        with open('data/%s/answer/%s|%s|%s' % (split, dataset, paragraph['id'], qa['id']), 'w') as f:
                            for answer in answers:
                                f.write('%s\n' % answer)
                        with open('data/%s/span/%s|%s|%s' % (split, dataset, paragraph['id'], qa['id']), 'w') as f:
                            f.write('%d %d' % (answer_start, answer_end))

                    parsed_count += 1
                    print('%s set: %d/%d (%.2f%%) \r' \
                          % (split, parsed_count, doc_count, 100*parsed_count/doc_count), end='')
            print()
            if bad_context > 0:
                print('%d bad contexts found.' % bad_context)
            if bad_question > 0:
                print('%d bad questions found.' % bad_question)
    exit(0)
