import sys
import numpy as np
from os.path import join

import torch
from transformers import BertTokenizer
from transformers import XLNetTokenizer, XLNetForQuestionAnswering

from utils import AdamW
from data import get_dataloader
from evaluate import f1_score, exact_match_score, metric_max_over_ground_truths

np.random.seed(42)
torch.manual_seed(42)

norm_tokenizer = BertTokenizer.from_pretrained('/home/M10815022/Models/bert-wwm-ext/')


def validate_dataset(model, split, tokenizer, dataset, topk=1):
    assert split in ('dev', 'test')
    fwd_dataloader = get_dataloader('xlnet', split, tokenizer, bwd=False, \
                        batch_size=16, num_workers=16, prefix=dataset)
    bwd_dataloader = get_dataloader('xlnet', split, tokenizer, bwd=True, \
                        batch_size=16, num_workers=16, prefix=dataset)
    em, f1, count = 0, 0, 0
    
    model.start_n_top = topk
    model.end_n_top = topk
    model.eval()
    for fwd_batch, bwd_batch in zip(fwd_dataloader, bwd_dataloader):
        # Forward
        input_ids, attention_mask, token_type_ids, fwd_input_tokens_no_unk, answers = fwd_batch
        input_ids = input_ids.cuda(device=device)
        attention_mask = attention_mask.cuda(device=device)
        token_type_ids = token_type_ids.cuda(device=device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        fwd_start_index = outputs[1]
        fwd_end_index = outputs[3].view(-1, model.end_n_top, model.start_n_top).permute([0,2,1])[:,:,0]
        
        # Backward
        input_ids, attention_mask, token_type_ids, bwd_input_tokens_no_unk, answers = bwd_batch
        input_ids = input_ids.cuda(device=device)
        attention_mask = attention_mask.cuda(device=device)
        token_type_ids = token_type_ids.cuda(device=device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bwd_start_index = outputs[1]
        bwd_end_index = outputs[3].view(-1, model.end_n_top, model.start_n_top).permute([0,2,1])[:,:,0] + 1
        
        for i, answer in enumerate(answers):
            preds = []
            for k in range(model.start_n_top):
                fwd_pred_tokens = fwd_input_tokens_no_unk[i][fwd_start_index[i][k]:fwd_end_index[i][k]]
                bwd_pred_tokens = bwd_input_tokens_no_unk[i][bwd_start_index[i][k]:bwd_end_index[i][k]]
                preds.append(tokenizer.convert_tokens_to_string(fwd_pred_tokens))
                preds.append(tokenizer.convert_tokens_to_string(bwd_pred_tokens))

            norm_preds_tokens = [norm_tokenizer.basic_tokenizer.tokenize(pred) for pred in preds]
            norm_preds = [norm_tokenizer.convert_tokens_to_string(norm_pred_tokens) for norm_pred_tokens in norm_preds_tokens]
            norm_answer_tokens = [norm_tokenizer.basic_tokenizer.tokenize(ans) for ans in answer]
            norm_answer = [norm_tokenizer.convert_tokens_to_string(ans_tokens) for ans_tokens in norm_answer_tokens]

            em += max(metric_max_over_ground_truths(exact_match_score, norm_pred, norm_answer) for norm_pred in norm_preds)
            f1 += max(metric_max_over_ground_truths(f1_score, norm_pred, norm_answer) for norm_pred in norm_preds)
            count += 1
    del fwd_dataloader, bwd_dataloader
    return em, f1, count

def validate(model, tokenizer, datasets, topk=1):
    val_sum_em, val_sum_f1, val_total_count = 0, 0, 0
    test_sum_em, test_sum_f1, test_total_count = 0, 0, 0
    for dataset in datasets:
        # Valid set
        val_em, val_f1, val_count = validate_dataset(model, 'dev', tokenizer, dataset, topk)
        val_sum_em += val_em
        val_sum_f1 += val_f1
        val_total_count += val_count
        val_avg_em = 100 * val_em / val_count
        val_avg_f1 = 100 * val_f1 / val_count
        
        # Test set
        test_em, test_f1, test_count = validate_dataset(model, 'test', tokenizer, dataset, topk)
        test_sum_em += test_em
        test_sum_f1 += test_f1
        test_total_count += test_count
        test_avg_em = 100 * test_em / test_count
        test_avg_f1 = 100 * test_f1 / test_count
        print('%s | val_em=%.5f, val_f1=%.5f | test_em=%.5f, test_f1=%.5f' \
            % (dataset, val_avg_em, val_avg_f1, test_avg_em, test_avg_f1))
    
    # Validate on all dataset
    val_avg_em = 100 * val_sum_em / val_total_count
    val_avg_f1 = 100 * val_sum_f1 / val_total_count
    test_avg_em = 100 * test_sum_em / test_total_count
    test_avg_f1 = 100 * test_sum_f1 / test_total_count
    print('%s | val_em=%.5f, val_f1=%.5f | test_em=%.5f, test_f1=%.5f' \
            % ('(ALL)', val_avg_em, val_avg_f1, test_avg_em, test_avg_f1))

    # BONUS: test on FGC samples
    test_em, test_f1, test_count = validate_dataset(model, 'test', tokenizer, 'FGC', topk)
    test_avg_em = 100 * test_em / test_count
    test_avg_f1 = 100 * test_f1 / test_count
    print('FGC | em=%.5f, f1=%.5f' % (test_avg_em, test_avg_f1))
    return val_avg_f1


if __name__ == '__main__':
    
    if len(sys.argv) < 5:
        print('Usage: python3 train_xlnet.py cuda:N <model_path> <save_path> <dataset_1> <dataset_2> ... <dataset_n>')
        exit(1)


    # Config
    lr = 2e-5
    batch_size = 4
    accumulate_batch_size = 32
    
    assert accumulate_batch_size % batch_size == 0
    update_stepsize = accumulate_batch_size // batch_size


    dataset = sys.argv[4:]
    model_path = sys.argv[2]
    tokenizer = XLNetTokenizer.from_pretrained(model_path)
    model = XLNetForQuestionAnswering.from_pretrained(model_path)

    device = torch.device(sys.argv[1])
    model.to(device)
    model.start_n_top = 5
    model.end_n_top = 5

    optimizer = AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad()

    step = 0
    patience, best_val = 0, 0
    best_state_dict = model.state_dict()
    dataloader = get_dataloader('xlnet', 'train', tokenizer, batch_size=batch_size, num_workers=16)

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
                val_f1 = validate(model, tokenizer, dataset, topk=5)
                if val_f1 > best_val:
                     patience = 0
                     best_val = val_f1
                     best_state_dict = model.state_dict()
                else:
                     patience += 1

            if patience > 5 or step >= 200000:
                print('Finish training.')
                save_path = join(sys.argv[3], 'finetune.ckpt')
                torch.save(best_state_dict, save_path)
                del model, dataloader
                exit(0)
