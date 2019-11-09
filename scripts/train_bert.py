import sys
import numpy as np
from os.path import join

import torch
from transformers import BertTokenizer, BertForQuestionAnswering

from utils import AdamW
from data import get_dataloader
from evaluate import f1_score, exact_match_score, metric_max_over_ground_truths

np.random.seed(42)
torch.manual_seed(42)


def validate_dataset(model, split, tokenizer, dataset):
    assert split in ('dev', 'test')
    dataloader = get_dataloader('bert', split, tokenizer, \
            batch_size=32, num_workers=16, prefix=dataset)
    em, f1, count = 0, 0, 0
    
    for j, batch in enumerate(dataloader):
        input_ids, attention_mask, token_type_ids, start_positions, \
            end_positions, input_tokens_no_unk, answers = batch
        
        input_ids = input_ids.cuda(device=device)
        attention_mask = attention_mask.cuda(device=device)
        token_type_ids = token_type_ids.cuda(device=device)
        start_positions = start_positions.cuda(device=device)
        end_positions = end_positions.cuda(device=device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        start_preds = outputs[1][:, 0]
        end_preds = outputs[3][:, 0] + 1
        count += len(answers)
        
        for i, answer in enumerate(answers):
            pred = input_tokens_no_unk[i][start_preds[i]:end_preds[i]]
            pred = tokenizer.convert_tokens_to_string(pred)
            em += metric_max_over_ground_truths(exact_match_score, pred, answer)
            f1 += metric_max_over_ground_truths(f1_score, pred, answer)
    del dataloader
    return em, f1, count

def validate(model, tokenizer, datasets):
    val_sum_em, val_sum_f1, val_total_count = 0, 0, 0
    test_sum_em, test_sum_f1, test_total_count = 0, 0, 0
    for dataset in datasets:
        # Valid set
        val_em, val_f1, val_count = validate_dataset(model, 'dev', tokenizer, dataset)
        val_sum_em += val_em
        val_sum_f1 += val_f1
        val_total_count += val_count
        val_avg_em = 100 * val_em / val_count
        val_avg_f1 = 100 * val_f1 / val_count
        
        # Test set
        test_em, test_f1, test_count = validate_dataset(model, 'test', tokenizer, dataset)
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
    test_em, test_f1, test_count = validate_dataset(model, 'test', tokenizer, 'FGC')
    test_avg_em = 100 * test_em / test_count
    test_avg_f1 = 100 * test_f1 / test_count
    print('FGC | em=%.5f, f1=%.5f' % (test_avg_em, test_avg_f1))
    return val_avg_f1


if __name__ == '__main__':
    
    if len(sys.argv) < 5:
        print('Usage: python3 train_bert.py cuda:N <model_path> <save_path> <dataset_1> <dataset_2> ... <dataset_n>')
        exit(1)


    # Config
    lr = 2e-5
    max_norm = 2.0
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
    model.start_n_top = 1
    model.end_n_top = 1

    optimizer = AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad()

    step = 0
    patience, best_val = 0, 0
    best_state_dict = model.state_dict()
    dataloader = get_dataloader('bert', 'train', tokenizer, batch_size=batch_size, num_workers=16)

    print('Start training...')
    while True:
        for batch in dataloader:
            input_ids, attention_mask, token_type_ids, start_positions, \
                end_positions, input_tokens_no_unk, answers = batch

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
                val_f1 = validate(model, tokenizer, dataset)
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
