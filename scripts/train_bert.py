import sys
import numpy as np
from os.path import join
from copy import deepcopy

import torch
from torch.nn.functional import softmax
from transformers import BertTokenizer, BertForQuestionAnswering

from utils import AdamW
from data import get_dataloader
from evaluate import f1_score, exact_match_score, metric_max_over_ground_truths

np.random.seed(42)
torch.manual_seed(42)

norm_tokenizer = BertTokenizer.from_pretrained('/home/M10815022/Models/bert-wwm-ext/')


def validate_dataset(model, split, tokenizer, topk=1):
    assert split in ('dev', 'test')
    dataloader = get_dataloader('bert', split, tokenizer, bwd=False, \
                        batch_size=16, num_workers=16)
    em, f1, count = 0, 0, 0
    
    model.eval()
    for batch in dataloader:
        input_ids, attention_mask, token_type_ids, input_tokens_no_unk, answers = batch
        input_ids = input_ids.cuda(device=device)
        attention_mask = attention_mask.cuda(device=device)
        token_type_ids = token_type_ids.cuda(device=device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            
        start_logits, end_logits = outputs[0], outputs[1]
        start_probs = softmax(start_logits, dim=1)
        end_probs = softmax(end_logits, dim=1)
        start_probs, start_index = start_probs.topk(topk, dim=1)
        end_probs, end_index = end_probs.topk(topk, dim=1)
        
        for i, answer in enumerate(answers):
            preds = []
            if topk <= 1:
                span_tokens = input_tokens_no_unk[i][start_index[i][0]:end_index[i][0] + 1]
                preds.append(tokenizer.convert_tokens_to_string(span_tokens))
            else:
                joint_probs, joint_index = (start_probs[i].unsqueeze(1) * end_probs[i].unsqueeze(0)).view(topk*topk).topk(topk)
                for n in range(topk):
                    smap = joint_index[n] // topk
                    emap = joint_index[n] - smap * topk
                    span_tokens = input_tokens_no_unk[i][start_index[i][smap]:end_index[i][emap] + 1]
                    preds.append(tokenizer.convert_tokens_to_string(span_tokens))
                
            norm_preds_tokens = [norm_tokenizer.basic_tokenizer.tokenize(pred) for pred in preds]
            norm_preds = [norm_tokenizer.convert_tokens_to_string(norm_pred_tokens) for norm_pred_tokens in norm_preds_tokens]
            norm_answer_tokens = [norm_tokenizer.basic_tokenizer.tokenize(ans) for ans in answer]
            norm_answer = [norm_tokenizer.convert_tokens_to_string(ans_tokens) for ans_tokens in norm_answer_tokens]

            em += max(metric_max_over_ground_truths(exact_match_score, norm_pred, norm_answer) for norm_pred in norm_preds)
            f1 += max(metric_max_over_ground_truths(f1_score, norm_pred, norm_answer) for norm_pred in norm_preds)
            count += 1
    del dataloader
    return em, f1, count

def validate(model, tokenizer, topk=1):    
    # Valid set
    val_em, val_f1, val_count = validate_dataset(model, 'dev', tokenizer, topk)
    val_avg_em = 100 * val_em / val_count
    val_avg_f1 = 100 * val_f1 / val_count

    # Test set
    test_em, test_f1, test_count = validate_dataset(model, 'test', tokenizer, topk)
    test_avg_em = 100 * test_em / test_count
    test_avg_f1 = 100 * test_f1 / test_count
    
    print('%d-best | val_em=%.5f, val_f1=%.5f | test_em=%.5f, test_f1=%.5f' \
        % (topk, val_avg_em, val_avg_f1, test_avg_em, test_avg_f1))
    return val_avg_f1


if __name__ == '__main__':
    
    if len(sys.argv) != 4:
        print('Usage: python3 train_bert.py cuda:<n> <model_path> <save_path>')
        exit(1)


    # Config
    lr = 2e-5
    batch_size = 4
    accumulate_batch_size = 32
    
    assert accumulate_batch_size % batch_size == 0
    update_stepsize = accumulate_batch_size // batch_size

    
    model_path = sys.argv[2]
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForQuestionAnswering.from_pretrained(model_path)

    device = torch.device(sys.argv[1])
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad()

    step = 0
    patience, best_val = 0, 0
    best_state_dict = model.state_dict()
    dataloader = get_dataloader('bert', 'train', tokenizer, batch_size=batch_size, num_workers=16)

    print('Start training...')
    while True:
        for batch in dataloader:
            input_ids, attention_mask, token_type_ids, start_positions, end_positions = batch

            input_ids = input_ids.cuda(device=device)
            attention_mask = attention_mask.cuda(device=device)
            token_type_ids = token_type_ids.cuda(device=device)
            start_positions = start_positions.cuda(device=device)
            end_positions = end_positions.cuda(device=device)
    
            model.train()
            loss = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, \
                               start_positions=start_positions, end_positions=end_positions)[0]
            loss.backward()
            step += 1
            print('step %d | Training...\r' % step, end='')   
            if step % update_stepsize == 0:
                optimizer.step()
                optimizer.zero_grad()
    
            if step % 3000 == 0:
                print("step %d | Validating..." % step)
                val_f1 = validate(model, tokenizer, topk=5)
                if val_f1 > best_val:
                    patience = 0
                    best_val = val_f1
                    best_state_dict = deepcopy(model.state_dict())
                else:
                    patience += 1

            if patience > 20 or step >= 200000:
                print('Finish training. Scoring 1-5 best results...')
                save_path = join(sys.argv[3], 'finetune.ckpt')
                torch.save(best_state_dict, save_path)
                model.load_state_dict(best_state_dict)
                for k in range(1, 6):
                    validate(model, tokenizer, topk=k)
                del model, dataloader
                exit(0)
