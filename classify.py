from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import sys
import json
import numpy as np

def train_loop(model, linear, optimizer, criterion, tokenizer, train, num_choices, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} ///////////////////////////////")

        train_len = len(train)
        total_loss = 0.0

        for train_i in range(train_len):
            observation = train[train_i]
            contexts = []
            labels = []
            mask = torch.zeros((4, 2), dtype=float).to(device)

            for choice_i in range(num_choices):
                context = observation[choice_i][0]
                label = observation[choice_i][1]
                contexts.append(context)
                labels.append(label)
                mask[choice_i][label] = 1
            
            inputs = tokenizer(contexts, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
            # inputs = inputs.to(device)
            inputs['input_ids'] = inputs['input_ids'].to(device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
            inputs['attention_mask'] = inputs['attention_mask'].to(device)

            optimizer.zero_grad()
            hidden = model(**inputs)

            logits = torch.matmul(hidden.last_hidden_state[:, 0, :], linear)

            probs = F.softmax(logits, dim=1)
            correct_probs = probs * mask
            log_probs = torch.log(torch.sum(correct_probs, dim=1))
            loss = -torch.sum(log_probs)
            # print(loss)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            
            if train_i % 1000 == 0:
                print(train_i, "/", train_len)
        
        average_loss = total_loss / train_len
        print(f"Average Loss: {average_loss}")

        torch.save(model.state_dict(), 'classify.pth')

def test_loop(model, linear, tokenizer, test, num_choices):
    test_len = len(test)
    running_accuracy = 0

    for test_i in range(test_len):
        observation = test[test_i]
        contexts = []
        labels = []
        mask = torch.zeros((4, 2), dtype=float).to(device) ##### if code doesnt work, change to numpy

        for choice_i in range(num_choices):
            context = observation[choice_i][0]
            label = observation[choice_i][1]
            contexts.append(context)
            labels.append(label)
            mask[choice_i][label] = 1
        
        inputs = tokenizer(contexts, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
        inputs = inputs.to(device)

        hidden = model(**inputs)

        logits = torch.matmul(hidden.last_hidden_state[:, 0, :], linear)
        probs = F.softmax(logits, dim=1)[:, 1]
        labels = torch.tensor(labels).to(device)

        # print(probs)
        # print(labels)

        pred = torch.argmax(probs)
        real = torch.argmax(labels)
        if pred == real:
            running_accuracy += 1
        
    average_accuracy = running_accuracy / test_len
    return average_accuracy
    # print(f"Average Accuracy: {average_accuracy}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():  
    torch.manual_seed(0)
    answers = ['A','B','C','D']

    train = []
    test = []
    valid = []
    
    file_name = 'train_complete.jsonl'        
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])
        
        obs = []
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text,label])
        train.append(obs)
        
        # print(obs)
        # print(' ')
        
        # print(result['question']['stem'])
        # print(' ',result['question']['choices'][0]['label'],result['question']['choices'][0]['text'])
        # print(' ',result['question']['choices'][1]['label'],result['question']['choices'][1]['text'])
        # print(' ',result['question']['choices'][2]['label'],result['question']['choices'][2]['text'])
        # print(' ',result['question']['choices'][3]['label'],result['question']['choices'][3]['text'])
        # print('  Fact: ',result['fact1'])
        # print('  Answer: ',result['answerKey'])
        # print('  ')
                
    file_name = 'dev_complete.jsonl'        
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])
        
        obs = []
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text,label])
        valid.append(obs)
        
    file_name = 'test_complete.jsonl'        
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])
        
        obs = []
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text,label])
        test.append(obs)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    criterion = nn.NLLLoss()
    linear = torch.rand(768, 2)
    
    model = model.to(device)
    linear = linear.to(device)

    # Add code to fine-tune and test your MCQA classifier.
    # Use to toggle between training and testing

    is_training = False
    is_zero_shot = False

    if is_training:
        train_loop(model, linear, optimizer, criterion, tokenizer, train, 4, 5)
    else:
        if not is_zero_shot:
            model.load_state_dict(torch.load('classify.pth'))
            model.eval()

        av_valid_acc = test_loop(model, linear, tokenizer, valid, 4)
        print(f"Valid Average Accuracy: {av_valid_acc}")

        av_test_acc = test_loop(model, linear, tokenizer, test, 4)
        print(f"Test Average Accuracy: {av_test_acc}")

if __name__ == "__main__":
    main()