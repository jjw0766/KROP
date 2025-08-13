import re
import torch

from collections import deque, Counter
from jamotools import split_syllables, join_jamos
from transformers import AutoTokenizer, Qwen2Tokenizer

class BonitaTokenizer:
    def __init__(self, base_tokenizer_name: Qwen2Tokenizer, bot_token='<im_start>', eot_token='<im_end>', eod_token='<endoftext>'):
        self.base_tokenizer_name = base_tokenizer_name
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_tokenizer_name)
        self.pad_token, self.pad_token_id = self.get_pad_token(self.base_tokenizer)
        self.bot_token = bot_token
        self.bot_token_id = self.base_tokenizer.encode(bot_token, add_special_tokens=False)[-1]
        self.eot_token = eot_token
        self.eot_token_id = self.base_tokenizer.encode(eot_token, add_special_tokens=False)[-1]
        self.eod_token = eod_token
        self.eod_token_id = self.base_tokenizer.encode(eod_token, add_special_tokens=False)[-1]
        self.space_token_id = self.base_tokenizer.encode(' ', add_special_tokens=False)[-1]

    def get_pad_token(self, tokenizer):
        max_len = -1
        max_len_token = None
        for token, token_id in tokenizer.vocab.items():
            if len(token) > max_len:
                max_len = len(token)
                max_len_token = token
        return max_len_token, tokenizer.vocab[max_len_token]

    def encode(self, sentence_noisy, sentence):
        # messages = [
        #     {"role": "user", "content": f"Task: Correct the text, Input: {sentence_noisy}"},
        # ]
        # prefix_ids = self.base_tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=True,
        #     add_generation_prompt=True,
        #     enable_thinking=False, # Switches between thinking and non-thinking modes. Default is True.
        # )
        prefix_ids = self.base_tokenizer.encode(f"Task: Correct the text, Input: {sentence_noisy}")

        sentence_noisy_ids = []
        for char in sentence_noisy:
            sentence_noisy_id = self.base_tokenizer.encode(char, add_special_tokens=False)
            sentence_noisy_id = sentence_noisy_id + (4-len(sentence_noisy_id)) * [self.pad_token_id]
            sentence_noisy_ids.extend(sentence_noisy_id)
        sentence_ids = []
        for char in sentence:
            sentence_id = self.base_tokenizer.encode(char, add_special_tokens=False)
            sentence_id = sentence_id + (4-len(sentence_id)) * [self.pad_token_id]
            sentence_ids.extend(sentence_id)

        encoded_ids = [self.bot_token_id] + prefix_ids + sentence_noisy_ids + [self.eot_token_id]
        label_ids = [-100] + [-100] * len(prefix_ids) + sentence_ids + [-100]
        return encoded_ids, label_ids

    def decode(self, encoded_ids, label_ids):
        sentence_ids = [encoded_id for encoded_id, label_id in zip(encoded_ids, label_ids) if label_id!=-100]

        sentence_ids = deque(sentence_ids)
        decoded = []
        past_ids = []
        while len(sentence_ids):
            sentence_id = sentence_ids.popleft()

            decoded.append(self.base_tokenizer.decode(past_ids))
            past_ids = []
            id1 = sentence_id
            id2 = sentence_ids.popleft()
            id3 = sentence_ids.popleft()
            id4 = sentence_ids.popleft()
            char = self.base_tokenizer.decode([id1, id2, id3, id4])[:1]
            decoded.append(char)
        decoded.append(self.base_tokenizer.decode(past_ids))
        return ''.join(decoded)

    def batch_encode(self, sentences_noisy, sentences):
        input_ids = []
        attention_mask = []
        label_ids = []
        for sentence_noisy, sentence in zip(sentences_noisy, sentences):
            input_ids_row, label_ids_row = self.encode(sentence_noisy, sentence)
            input_ids.append(input_ids_row)
            label_ids.append(label_ids_row)
        max_length = max(list(map(len, input_ids)))
        for i in range(len(sentences)):
            input_ids[i] = input_ids[i] + (max_length-len(input_ids[i])) * [self.eod_token_id]
            label_ids[i] = label_ids[i] + (max_length-len(label_ids[i])) * [-100]
            attention_mask.append([1 if input_id!=self.eod_token_id else 0 for input_id in input_ids[i]])
        return (
            torch.LongTensor(input_ids),
            torch.LongTensor(label_ids),
            torch.LongTensor(attention_mask),
        )

class BINDKTokenizer:    
    def __init__(self, base_tokenizer_name):
        self.base_tokenizer_name = base_tokenizer_name
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_tokenizer_name)
        self.base_tokenizer.bos_token_id = self.base_tokenizer.encode('<|im_start|>', add_special_tokens=False)[-1]
        self.base_tokenizer.bos_token = '<|im_start|>'
        self.base_tokenizer.eos_token_id = self.base_tokenizer.encode('<|im_end|>', add_special_tokens=False)[-1]
        self.base_tokenizer.eos_token = '<|im_end|>'
        self.base_tokenizer.pad_token_id = 140783
        self.pad_token_id = 140783
        self.base_tokenizer.pad_token = self.base_tokenizer.decode(self.pad_token_id)
        self.space_token_id = self.base_tokenizer.encode(' ', add_special_tokens=False)[-1]
        char_start, char_end = 0xAC00, 0xD7A3  # 가-힣
        self.kor_chars = list(set([chr(code) for code in range(char_start, char_end + 1)]))
        self.char_ids = []
        for kor_char in self.kor_chars:
            ids = self.base_tokenizer.encode(kor_char, add_special_tokens=False)
            if len(ids)==1:
                ids = ids+2*[self.pad_token_id]
                self.char_ids.append(ids)
            elif len(ids)==2:
                ids = ids+[self.pad_token_id]
                self.char_ids.append(ids)
            elif len(ids)==3:
                ids = ids
                self.char_ids.append(ids)
        self.chos = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.joongs = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        self.jongs = [self.base_tokenizer.pad_token, 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        jamos = list(set(self.chos) | set(self.joongs) | set(self.jongs))
        jamo_ids = self.base_tokenizer(jamos, add_special_tokens=False)['input_ids']
        self.jamo_to_id = {jamo: jamo_id[-1] for jamo, jamo_id in zip(jamos, jamo_ids)}
        self.cho_ids = [self.jamo_to_id[cho] for cho in self.chos]
        self.joong_ids = [self.jamo_to_id[joong] for joong in self.joongs]
        self.jong_ids = [self.jamo_to_id[jong] for jong in self.jongs]
        self.id_to_jamo = {jamo_id: jamo for jamo, jamo_id in self.jamo_to_id.items()}

    def encode_jamo(self, sentence):
        sentence = self.base_tokenizer.bos_token + sentence + self.base_tokenizer.eos_token
        encoded_ids = []
        token_type_ids = []
        past_chars = ''
        for char in sentence:
            if char in self.kor_chars:
                if past_chars:
                    past_chars_encoded = self.base_tokenizer.encode(past_chars, add_special_tokens=False)
                    encoded_ids.extend(past_chars_encoded)
                    token_type_ids.extend([0]*len(past_chars_encoded))
                past_chars=''
                char_splitted = list(split_syllables(char))[:3]
                char_splitted = char_splitted + (3-len(char_splitted))*[self.base_tokenizer.pad_token]
                cho, joong, jong = char_splitted
                encoded_ids.extend([self.jamo_to_id[cho], self.jamo_to_id[joong], self.jamo_to_id[jong]])
                token_type_ids.extend([1,2,3])
            else:
                past_chars = past_chars+char
        if past_chars:
            past_chars_encoded = self.base_tokenizer.encode(past_chars, add_special_tokens=False)
            encoded_ids.extend(past_chars_encoded)
            token_type_ids.extend([0]*len(past_chars_encoded))
        return encoded_ids, token_type_ids

    def decode_jamo(self, encoded_ids, token_type_ids, contain_bos_eos_token=True):
        if contain_bos_eos_token:
            encoded_ids = encoded_ids[1:-1]
            token_type_ids = token_type_ids[1:-1]

        encoded_ids = deque(encoded_ids)
        token_type_ids = deque(token_type_ids)
        decoded = []
        past_ids = []
        while len(encoded_ids):
            encoded_id = encoded_ids.popleft()
            token_type_id = token_type_ids.popleft()
            if token_type_id==0:
                past_ids.append(encoded_id)
            else:
                decoded.append(self.base_tokenizer.decode(past_ids))
                past_ids = []
                cho_id = encoded_id
                joong_id = encoded_ids.popleft()
                jong_id = encoded_ids.popleft()
                token_type_ids.popleft()
                token_type_ids.popleft()
                char = join_jamos([self.id_to_jamo[cho_id], self.id_to_jamo[joong_id], self.id_to_jamo[jong_id]])[:1]
                decoded.append(char)
        decoded.append(self.base_tokenizer.decode(past_ids))
        return ''.join(decoded)

    def encode_char(self, sentence):
        sentence = self.base_tokenizer.bos_token + sentence + self.base_tokenizer.eos_token
        encoded_ids = []
        token_type_ids = []
        past_chars = ''
        for char in sentence:
            if char in self.kor_chars:
                if past_chars:
                    past_chars_encoded = self.base_tokenizer.encode(past_chars, add_special_tokens=False)
                    encoded_ids.extend(past_chars_encoded)
                    token_type_ids.extend([0]*len(past_chars_encoded))
                past_chars=''
                encoded_id = self.base_tokenizer.encode(char, add_special_tokens=False)
                encoded_id = encoded_id + (3-len(encoded_id)) * [self.pad_token_id]
                encoded_ids.extend(encoded_id)
                token_type_ids.extend([4,4,4])
            else:
                past_chars = past_chars+char
        if past_chars:
            past_chars_encoded = self.base_tokenizer.encode(past_chars, add_special_tokens=False)
            encoded_ids.extend(past_chars_encoded)
            token_type_ids.extend([0]*len(past_chars_encoded))
        return encoded_ids, token_type_ids

    def decode_char(self, encoded_ids, token_type_ids, contain_bos_eos_token=True):
        if contain_bos_eos_token:
            encoded_ids = encoded_ids[1:-1]
            token_type_ids = token_type_ids[1:-1]

        encoded_ids = deque(encoded_ids)
        token_type_ids = deque(token_type_ids)
        decoded = []
        past_ids = []
        while len(encoded_ids):
            encoded_id = encoded_ids.popleft()
            token_type_id = token_type_ids.popleft()
            if token_type_id==0:
                past_ids.append(encoded_id)
            else:
                decoded.append(self.base_tokenizer.decode(past_ids))
                past_ids = []
                id1 = encoded_id
                id2 = encoded_ids.popleft()
                id3 = encoded_ids.popleft()
                token_type_ids.popleft()
                token_type_ids.popleft()
                [id1, id2, id3]
                char = self.base_tokenizer.decode([id1, id2, id3])[:1]
                decoded.append(char)
        decoded.append(self.base_tokenizer.decode(past_ids))
        return ''.join(decoded)

    def encode_jamo_from_char_encoded(self, encoded_ids, token_type_ids):
        encoded_ids = deque(encoded_ids)
        token_type_ids = deque(token_type_ids)
        encoded_ids_new = []
        token_type_ids_new = []
        while len(encoded_ids):
            encoded_id = encoded_ids.popleft()
            token_type_id = token_type_ids.popleft()
            if token_type_id==0:
                encoded_ids_new.append(encoded_id)
                token_type_ids_new.append(token_type_id)
            else:
                encoded_id2 = encoded_ids.popleft()
                encoded_id3 = encoded_ids.popleft()
                token_type_ids.popleft()
                token_type_ids.popleft()
                char = self.base_tokenizer.decode([encoded_id, encoded_id2, encoded_id3])[0]
                char_splitted = list(split_syllables(char))[:3]
                char_splitted = char_splitted + (3-len(char_splitted))*[self.base_tokenizer.pad_token]
                cho, joong, jong = char_splitted
                encoded_ids_new.extend([self.jamo_to_id[cho], self.jamo_to_id[joong], self.jamo_to_id[jong]])
                token_type_ids_new.extend([1,2,3])
        return encoded_ids_new, token_type_ids_new

    def batch_encode_char(self, sentences):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        for sentence in sentences:
            input_ids_row, token_type_id = self.encode_char(sentence)
            input_ids.append(input_ids_row)
            token_type_ids.append(token_type_id)
        max_length = max(list(map(len, input_ids)))
        for i in range(len(sentences)):
            input_ids[i] = input_ids[i] + (max_length-len(input_ids[i])) * [self.base_tokenizer.eos_token_id]
            attention_mask.append([1 if input_id!=self.base_tokenizer.eos_token_id else 0 for input_id in input_ids[i]])
            token_type_ids[i] = token_type_ids[i] + (max_length-len(token_type_ids[i])) * [0]
        return (
            torch.LongTensor(input_ids),
            torch.LongTensor(attention_mask),
            torch.LongTensor(token_type_ids)
        )

    def batch_encode_jamo_from_char_encoded(self, batch_encoded_ids, batch_token_type_ids):
        input_ids = []
        attention_mask = []
        token_type_ids_new = []
        for encoded_ids, token_type_ids in zip(batch_encoded_ids, batch_token_type_ids):
            encoded_ids_row, token_type_ids_row = self.encode_jamo_from_char_encoded(encoded_ids, token_type_ids)
            attention_mask.append([1 if encoded_id!=self.base_tokenizer.eos_token_id else 0 for encoded_id in encoded_ids_row])
            input_ids.append(encoded_ids_row)
            token_type_ids_new.append(token_type_ids_row)
        
        return (
            torch.LongTensor(input_ids),
            torch.LongTensor(attention_mask),
            torch.LongTensor(token_type_ids_new),
        )

    def batch_encode_jamo(self, sentences):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        for sentence in sentences:
            input_ids_row, token_type_id = self.encode_jamo(sentence)
            input_ids.append(input_ids_row)
            token_type_ids.append(token_type_id)
        max_length = max(list(map(len, input_ids)))
        
        for i in range(len(sentences)):
            input_ids[i] = input_ids[i] + (max_length-len(input_ids[i])) * [self.base_tokenizer.eos_token_id]
            attention_mask.append([1 if input_id!=self.base_tokenizer.eos_token_id else 0 for input_id in input_ids[i]])
            token_type_ids[i] = token_type_ids[i] + (max_length-len(token_type_ids[i])) * [0]
            token_type_ids[i][0] = 0

        return (
            torch.LongTensor(input_ids),
            torch.LongTensor(attention_mask),
            torch.LongTensor(token_type_ids),
        )
    
class BINDCTokenizer:    
    def __init__(self, base_tokenizer_name):
        self.base_tokenizer_name = base_tokenizer_name
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_tokenizer_name)
        self.base_tokenizer.bos_token_id = self.base_tokenizer.encode('<|im_start|>', add_special_tokens=False)[-1]
        self.base_tokenizer.bos_token = '<|im_start|>'
        self.base_tokenizer.eos_token_id = self.base_tokenizer.encode('<|im_end|>', add_special_tokens=False)[-1]
        self.base_tokenizer.eos_token = '<|im_end|>'
        self.base_tokenizer.pad_token_id = 140783
        self.pad_token_id = 140783
        self.base_tokenizer.pad_token = self.base_tokenizer.decode(self.pad_token_id)
        self.space_token_id = self.base_tokenizer.encode(' ', add_special_tokens=False)[-1]
        hanzi_ranges = [
            (0x4E00, 0x9FFF),       # 기본 한자
            (0x3400, 0x4DBF),       # 확장 A
            (0xF900, 0xFAFF),       # 호환 한자
            (0x20000, 0x2A6DF),     # 확장 B
            (0x2A700, 0x2B73F),     # 확장 C
            (0x2B740, 0x2B81F),     # 확장 D
            (0x2B820, 0x2CEAF),     # 확장 E
            (0x2CEB0, 0x2EBEF),     # 확장 F
        ]

        # 모든 범위를 합쳐서 리스트 생성
        self.hanzi_chars = list(
            set(
                chr(code)
                for start, end in hanzi_ranges
                for code in range(start, end + 1)
            )
        )

        self.char_ids = []
        for hanzi_char in self.hanzi_chars:
            ids = self.base_tokenizer.encode(hanzi_char, add_special_tokens=False)
            if len(ids)==1:
                ids = ids+3*[self.pad_token_id]
                self.char_ids.append(ids)
            elif len(ids)==2:
                ids = ids+2*[self.pad_token_id]
                self.char_ids.append(ids)
            elif len(ids)==3:
                ids = ids+[self.pad_token_id]
                self.char_ids.append(ids)
            else:
                ids = ids
                self.char_ids.append(ids)

    def encode_char(self, sentence):
        sentence = self.base_tokenizer.bos_token + sentence + self.base_tokenizer.eos_token
        encoded_ids = []
        token_type_ids = []
        past_chars = ''
        for char in sentence:
            if char in self.hanzi_chars:
                if past_chars:
                    past_chars_encoded = self.base_tokenizer.encode(past_chars, add_special_tokens=False)
                    encoded_ids.extend(past_chars_encoded)
                    token_type_ids.extend([0]*len(past_chars_encoded))
                past_chars=''
                encoded_id = self.base_tokenizer.encode(char, add_special_tokens=False)
                encoded_id = encoded_id + (4-len(encoded_id)) * [self.pad_token_id]
                encoded_ids.extend(encoded_id)
                token_type_ids.extend([1,1,1,1])
            else:
                past_chars = past_chars+char
        if past_chars:
            past_chars_encoded = self.base_tokenizer.encode(past_chars, add_special_tokens=False)
            encoded_ids.extend(past_chars_encoded)
            token_type_ids.extend([0]*len(past_chars_encoded))
        return encoded_ids, token_type_ids

    def decode_char(self, encoded_ids, token_type_ids, contain_bos_eos_token=True):
        if contain_bos_eos_token:
            encoded_ids = encoded_ids[1:-1]
            token_type_ids = token_type_ids[1:-1]

        encoded_ids = deque(encoded_ids)
        token_type_ids = deque(token_type_ids)
        decoded = []
        past_ids = []
        while len(encoded_ids):
            encoded_id = encoded_ids.popleft()
            token_type_id = token_type_ids.popleft()
            if token_type_id==0:
                past_ids.append(encoded_id)
            else:
                decoded.append(self.base_tokenizer.decode(past_ids))
                past_ids = []
                id1 = encoded_id
                id2 = encoded_ids.popleft()
                id3 = encoded_ids.popleft()
                id4 = encoded_ids.popleft()
                token_type_ids.popleft()
                token_type_ids.popleft()
                token_type_ids.popleft()
                char = self.base_tokenizer.decode([id1, id2, id3, id4])[:1]
                decoded.append(char)
        decoded.append(self.base_tokenizer.decode(past_ids))
        return ''.join(decoded)

    def batch_encode_char(self, sentences):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        for sentence in sentences:
            input_ids_row, token_type_id = self.encode_char(sentence)
            input_ids.append(input_ids_row)
            token_type_ids.append(token_type_id)
        max_length = max(list(map(len, input_ids)))
        for i in range(len(sentences)):
            input_ids[i] = input_ids[i] + (max_length-len(input_ids[i])) * [self.base_tokenizer.eos_token_id]
            attention_mask.append([1 if input_id!=self.base_tokenizer.eos_token_id else 0 for input_id in input_ids[i]])
            token_type_ids[i] = token_type_ids[i] + (max_length-len(token_type_ids[i])) * [0]
        return (
            torch.LongTensor(input_ids),
            torch.LongTensor(attention_mask),
            torch.LongTensor(token_type_ids)
        )
    
class BINDTokenizer:    
    def __init__(self, base_tokenizer_name):
        self.base_tokenizer_name = base_tokenizer_name
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_tokenizer_name)
        self.base_tokenizer.bos_token_id = self.base_tokenizer.encode('<|im_start|>', add_special_tokens=False)[-1]
        self.base_tokenizer.bos_token = '<|im_start|>'
        self.base_tokenizer.eos_token_id = self.base_tokenizer.encode('<|im_end|>', add_special_tokens=False)[-1]
        self.base_tokenizer.eos_token = '<|im_end|>'
        self.base_tokenizer.pad_token_id = 140783
        self.pad_token_id = 140783
        self.base_tokenizer.pad_token = self.base_tokenizer.decode(self.pad_token_id)
        self.space_token_id = self.base_tokenizer.encode(' ', add_special_tokens=False)[-1]

    def encode_char(self, sentence, add_bos_eos_token=True):
        encoded_ids = []
        for char in sentence:
            encoded_id = self.base_tokenizer.encode(char, add_special_tokens=False)
            encoded_id = encoded_id + (4-len(encoded_id)) * [self.pad_token_id]
            encoded_ids.extend(encoded_id)
        if add_bos_eos_token:
            encoded_ids = [self.base_tokenizer.bos_token_id] + encoded_ids + [self.base_tokenizer.eos_token_id]
        return encoded_ids

    def decode_char(self, encoded_ids, contain_bos_eos_token=True):
        if contain_bos_eos_token:
            encoded_ids = encoded_ids[1:-1]

        encoded_ids = deque(encoded_ids)
        decoded = []
        past_ids = []
        while len(encoded_ids):
            encoded_id = encoded_ids.popleft()

            decoded.append(self.base_tokenizer.decode(past_ids))
            past_ids = []
            id1 = encoded_id
            id2 = encoded_ids.popleft()
            id3 = encoded_ids.popleft()
            id4 = encoded_ids.popleft()
            char = self.base_tokenizer.decode([id1, id2, id3, id4])[:1]
            decoded.append(char)
        decoded.append(self.base_tokenizer.decode(past_ids))
        return ''.join(decoded)

    def batch_encode_char(self, sentences):
        input_ids = []
        attention_mask = []
        for sentence in sentences:
            input_ids_row = self.encode_char(sentence)
            input_ids.append(input_ids_row)
        max_length = max(list(map(len, input_ids)))
        for i in range(len(sentences)):
            input_ids[i] = input_ids[i] + (max_length-len(input_ids[i])) * [self.base_tokenizer.eos_token_id]
            attention_mask.append([1 if input_id!=self.base_tokenizer.eos_token_id else 0 for input_id in input_ids[i]])
        return (
            torch.LongTensor(input_ids),
            torch.LongTensor(attention_mask),
        )
    
class CharEncoderTokenizer:
    def __init__(self, base_tokenizer_name, space_token, unk_token, pad_token):
        self.base_tokenizer_name = base_tokenizer_name
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_tokenizer_name)
        self.space_token = space_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.space_token_id = self.base_tokenizer.encode(space_token, add_special_tokens=False)

    def encode_char(self, sentence):
        sentence = sentence.replace(' ', self.space_token)
        encoded_ids = []
        for char in sentence:
            encoded_id = self.base_tokenizer.encode(char, add_special_tokens=False)
            encoded_ids.extend(encoded_id)
        return encoded_ids
    
    def decode_char(self, encoded_ids, sentence=None):
        encoded_ids = deque(encoded_ids)
        decoded = []
        while len(encoded_ids):
            encoded_id = encoded_ids.popleft()
            char = self.base_tokenizer.decode([encoded_id])
            decoded.append(char)
        decoded = ''.join(decoded).replace(self.space_token, ' ')
        if sentence:
            decoded = decoded.replace(self.unk_token, 'ㆀ').replace(self.pad_token, 'ㆀ')
            if len(sentence)!=len(decoded):
                print(sentence)
                print(decoded)
            temp = []
            for idx, char_decoded in enumerate(decoded):
                if char_decoded=='ㆀ':
                    temp.append(sentence[idx])
                else:
                    temp.append(char_decoded)
            decoded = ''.join(temp)
        return decoded
    
    def batch_encode_char(self, sentences):
        input_ids = []
        attention_mask = []
        for sentence in sentences:
            input_ids_row = self.encode_char(sentence)
            input_ids.append(input_ids_row)
        max_length = max(list(map(len, input_ids)))
        for i in range(len(sentences)):
            input_ids[i] = input_ids[i] + (max_length-len(input_ids[i])) * [self.base_tokenizer.pad_token_id]
            attention_mask.append([1 if input_id!=self.base_tokenizer.pad_token_id else 0 for input_id in input_ids[i]])
        return (
            torch.LongTensor(input_ids),
            torch.LongTensor(attention_mask)
        )
    

class SentenceTokenizer:
    def __init__(
        self,
        min_length=32,
        max_length=64,
        n_overlap=3,
        roll=False
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.n_overlap = n_overlap
        self.roll = roll

    def split_text_into_sentences(self, text):
        split_text = text.split(' ')
        split_text = [split_text[i] + split_text[i + 1] for i in range(0, len(split_text) - 1, 2)] + ([split_text[-1]] if len(split_text) % 2 != 0 else [])

        return split_text

    def merge_chunks(self, chunks):
        merged_chunks = []
        buffer = ""

        for chunk in chunks:
            buffer += chunk
            if len(buffer) > self.min_length:  # If buffer meets the min length, finalize it
                merged_chunks.append(buffer)
                buffer = ""

        # Add any remaining buffer as the last chunk
        if buffer:
            merged_chunks.append(buffer)

        return merged_chunks

    def merge_chunks_reverse(self, chunks):
        chunks_reverse = []
        for chunk in chunks[::-1]:
            chunks_reverse.append(chunk[::-1])
            
        merged_chunks = []
        buffer = ""

        for chunk in chunks_reverse:
            buffer += chunk
            if len(buffer) > self.min_length:  # If buffer meets the min length, finalize it
                merged_chunks.append(buffer)
                buffer = ""

        # Add any remaining buffer as the last chunk
        if buffer:
            merged_chunks.append(buffer)

        res_merged_chunks = []
        for chunk in merged_chunks[::-1]:
            res_merged_chunks.append(chunk[::-1])

        return res_merged_chunks
        
    def split_text(self, text):
        words = self.split_space(text)
        
        # Step 2: Greedily merge words until the length of the merged text is shorter than max_length
        splitted_chunks = []
        buffer = []
        
        for word in words:
            buffer.append(word)  # Add the word to the buffer
            merged_text = ''.join(buffer)
            
            # If the merged text exceeds max_length, push the current buffer to the result
            if len(merged_text) > self.max_length:
                # Remove the last added word and save the current buffer as a chunk
                buffer.pop()
                splitted_chunks.append(''.join(buffer))
                buffer = [''+word]  # Start a new buffer with the current word
    
        # Step 3: Append the left over buffer
        if buffer:
            splitted_chunks.append(''.join(buffer))
    
        return splitted_chunks

    def tokenize(self, text):
        splitted_chunks = []
        # Step 1: Split text into sentences
        sentences = self.split_text_into_sentences(text)
        for chunk in sentences:
            if len(chunk)>=self.max_length:
                splitted_chunks.extend(self.split_text(chunk))
            else:
                splitted_chunks.append(chunk)
        merged_chunks = self.merge_chunks(splitted_chunks)
        merged_chunks = self.merge_chunks_reverse(merged_chunks)

        return merged_chunks

    def split_space(self, text):
        split_text = re.split(r'(\s+)', text)  # Keep spaces as part of the split parts
        filtered_text = [s + sp for s, sp in zip(split_text[::2], split_text[1::2] + [''])] 
        return filtered_text
    
    def overlap(self, chunks):
        if not chunks:
            return []
        if self.roll:
            chunks = [chunks[-1]] + chunks + [chunks[0]]
        res = []
        total_idx = 0
        for chunk_idx in range(len(chunks)-1):
            chunk_a, chunk_b = chunks[chunk_idx], chunks[chunk_idx+1]
            chunk_a_words, chunk_b_words = self.split_space(chunk_a), self.split_space(chunk_b)
            chunk_a_overlap_length, chunk_b_overlap_length = len(chunk_a_words)//self.n_overlap, len(chunk_b_words)//self.n_overlap
            for overlap_idx in range(self.n_overlap):
                chunk_a_past, chunk_a_overlap, chunk_b_overlap = ''.join(chunk_a_words[:chunk_a_overlap_length*overlap_idx]), ''.join(chunk_a_words[chunk_a_overlap_length*overlap_idx:]), ''.join(chunk_b_words[:chunk_b_overlap_length*overlap_idx])
                overlap = chunk_a_overlap+chunk_b_overlap
                start = total_idx+len(chunk_a_past)
                end = start + len(overlap)
                res.append((start, end, overlap))
            total_idx += len(chunk_a)
        res.append((total_idx, total_idx+len(chunks[-1]), chunks[-1]))

        return res

    def decode_overlap(self, chunks):
        if not chunks:
            return ""
        
        # Determine total length based on the largest end index
        max_length = max(end for _, end, _ in chunks)
        
        # Dictionary to store characters at each index
        index_char_map = {i: [] for i in range(max_length)}
        
        # Populate index_char_map with characters from chunks
        for start, end, chunk in chunks:
            for i, char in enumerate(chunk):
                index = start + i
                if index < max_length:
                    index_char_map[index].append(char)
        
        # Reconstruct text using majority vote
        reconstructed_text = []
        for i in range(max_length):
            most_common_char, _ = Counter(index_char_map[i]).most_common(1)[0]
            reconstructed_text.append(most_common_char)
        res = "".join(reconstructed_text)
        if self.roll:
            res = res[len(chunks[0][2]):-len(chunks[-1][2])]
    
        return res