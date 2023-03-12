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

def train_loop(model, optimizer, tokenizer, train, num_choices, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} ///////////////////////////////")

        train_len = len(train)
        total_loss = 0.0

        for train_i in range(train_len):
            observation = train[train_i]
            
            inputs = tokenizer(observation, return_tensors="pt")
            inputs = inputs.to(device)
            outputs = model(**inputs, labels=inputs['input_ids'])

            optimizer.zero_grad()
            loss = outputs[0]
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            
            if train_i % 1000 == 0:
                print(train_i, "/", train_len)
        
        average_loss = total_loss / train_len
        print(f"Average Loss: {average_loss}")

        torch.save(model.state_dict(), 'generate.pth')

def max_choice(tokens):
  try:
    a_index = tokens.index('A')
  except ValueError :
    a_index = -float('inf')
  try:
    b_index = tokens.index('B')
  except ValueError :
    b_index = -float('inf')
  try:
    c_index = tokens.index('C')
  except ValueError :
    c_index = -float('inf')
  try:
    d_index = tokens.index('D')
  except ValueError :
    d_index = -float('inf')
  #print(a_index, b_index, c_index, d_index)
  return len(tokens) - min(a_index, b_index, c_index, d_index) - 1

def test_loop(model, tokenizer, test):
    test_len = len(test)
    running_accuracy = 0

    for test_i in range(test_len):
        observation = test[test_i]
        
        inputs = tokenizer(observation[:-1], return_tensors="pt")
        inputs = inputs.to(device)
        outputs = model.generate(**inputs, max_length=len(inputs['input_ids'][0]) + 5, pad_token_id=tokenizer.eos_token_id)

        tokens = tokenizer.decode(outputs[0], skip_special_tokens=True).split(' ')
        
        pred = tokens[-1]
        real = observation[-1]
        # print('PRED: ', pred)
        # print('REAL: ', real)
        # print('OBS: ', observation)
        # print('TOKEN: ', tokens)
        if pred == real:
            running_accuracy += 1

        if test_i % 100 == 0:
                print(test_i, "/", test_len)
                if test_i != 0:
                  print(running_accuracy / test_i)
        
    average_accuracy = running_accuracy / test_len
    return average_accuracy

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
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem'] + ' [SEP] '
        
        for j in range(4):
            base = base + result['question']['choices'][j]['label'] + ' ' + result['question']['choices'][j]['text'] + ' '
            
        base = base + ' [SEP] ' + result['answerKey']
        train.append(base)
        
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
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem'] + ' [SEP] '
        
        for j in range(4):
            base = base + result['question']['choices'][j]['label'] + ' ' + result['question']['choices'][j]['text'] + ' '
            
        base = base + ' [SEP] ' + result['answerKey']
        valid.append(base)
        
    file_name = 'test_complete.jsonl'        
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = result['fact1'] + ' [SEP] ' + result['question']['stem'] + ' [SEP] '
        
        for j in range(4):
            base = base + result['question']['choices'][j]['label'] + ' ' + result['question']['choices'][j]['text'] + ' '
            
        base = base + ' [SEP] ' + result['answerKey']
        test.append(base)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    
    model = model.to(device)

    # Add code to fine-tune and test your MCQA classifier.
    # Use to toggle between training and testing

    is_training = False
    is_zero_shot = False

    if is_training:
        train_loop(model, optimizer, tokenizer, train, 4, 5)
    else:
        if not is_zero_shot:
            model.load_state_dict(torch.load('generate.pth'))
            model.eval()

        av_valid_acc = test_loop(model, tokenizer, valid)
        print(f"Valid Average Accuracy: {av_valid_acc}")

        av_test_acc = test_loop(model, tokenizer, test)
        print(f"Test Average Accuracy: {av_test_acc}")

if __name__ == "__main__":
    main()