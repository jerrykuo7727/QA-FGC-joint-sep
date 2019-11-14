import sys
import json
from os.path import join, exists
from transformers import XLNetTokenizer

def find_sublist_xlnet(a, b, tokenizer):
    str_b = tokenizer.convert_tokens_to_string(b)
    for i in range(len(a)-len(b)+1):
        str_a = tokenizer.convert_tokens_to_string(a[i:i+len(b)])
        if str_a == str_b:
            return i
    return -1


if __name__ == '__main__':
    
    if len(sys.argv) < 3:
        print('Usage: python3 prepare_xlnet_data.py <pretrained_model> <dataset_1> <dataset_2> ... <dataset_n>')
        exit(0)

    model_path = sys.argv[1]
    datasets = sys.argv[2:]
    tokenizer = XLNetTokenizer.from_pretrained(model_path)
    
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
            impossible_question = 0
            doc_count = len([paragraph for data in datas[split]['data'] for paragraph in data['paragraphs']])
            parsed_count = 0

            for data in datas[split]['data']:
                for paragraph in data['paragraphs']:

                    # Document
                    raw_context = paragraph['context']
                    context = tokenizer.tokenize(raw_context)
                    context_no_unk = context

                    # QA pairs
                    qas = []
                    for qa in paragraph['qas']:
                        processed_qa = {}
                        raw_question = qa['question']
                        question = tokenizer.tokenize(raw_question)
                        question_no_unk = question

                        raw_answers = [ans['text'] for ans in qa['answers']]
                        answer_no_unk = tokenizer.tokenize(raw_answers[0])
                        if not answer_no_unk:
                            impossible_question += 1
                            continue
                            
                        if answer_no_unk[0] == '▁':
                            answer_no_unk = answer_no_unk[1:]
                        answer_start = find_sublist_xlnet(context_no_unk, answer_no_unk, tokenizer)
                        answer_end = answer_start + len(answer_no_unk) - 1 if answer_start >= 0 else -1
                        if answer_start < 0:
                            impossible_question += 1

                        if answer_start >= 0 or split != 'train':
                            processed_qa['question'] = question
                            processed_qa['question_no_unk'] = question_no_unk
                            processed_qa['answer'] = raw_answers
                            processed_qa['answer_start'] = answer_start
                            processed_qa['answer_end'] = answer_end
                            processed_qa['id'] = qa['id']
                            qas.append(processed_qa)

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
            print('impossible_questions: %d' % impossible_question)
    exit(0)
