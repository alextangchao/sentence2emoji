from transformers import T5Tokenizer, T5ForConditionalGeneration
import csv
import emoji

# MODEL = "t5-small"
MODEL = ".\\output\\"
FILE_PATH = ".\\..\\490A final project data - Emoji-50-467.csv"
SENTENCE = 'senetence'
# LABEL = 'label'
LABEL = 'translate'


def read_csv(filepath):
    with open(filepath, 'r', encoding='utf8') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        sentences = []
        labels = []
        for row in csv_reader:
            if row[LABEL] != '':
                try:
                    sentences.append(row[SENTENCE])
                    current_label = [emoji.UNICODE_EMOJI['en'][c] for c in row[LABEL] if c in emoji.UNICODE_EMOJI['en']]
                    labels.append(current_label)
                    # print(current_label)
                except IndexError:
                    print(row)

    return sentences, labels


def tanslate_one_sentence(sentence):
    tokenizer = T5Tokenizer.from_pretrained(MODEL)
    model = T5ForConditionalGeneration.from_pretrained(MODEL)

    # print(tokenizer.add_tokens([":skull:"]))

    encoding = tokenizer(f'translate English to Emoji: {sentence}', return_tensors='pt')
    print(encoding)
    print(tokenizer.tokenize(f'translate English to Emoji: {sentence}'))
    outputs = model.generate(encoding["input_ids"])
    print(outputs)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def tanslate_sentence_list(sentence_list):
    tokenizer = T5Tokenizer.from_pretrained(MODEL, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(MODEL, local_files_only=True)

    # when generating, we will use the logits of right-most token to predict the next token
    # so the padding should be on the left
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # to avoid an error

    task_prefix = 'translate English to Emoji: '
    sentences = ['The house is wonderful.', 'I like to work in NYC.']  # use different length sentences to test batching
    inputs = tokenizer([task_prefix + sentence for sentence in sentence_list], return_tensors="pt", padding=True)

    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        do_sample=False,  # disable sampling to test if batching affects output
    )

    return tokenizer.batch_decode(output_sequences, skip_special_tokens=True)


<<<<<<< HEAD:Models/T5/T5_test.py
sentences, labels = read_csv(FILE_PATH)
tanslate_one_sentence("Nice to see you!")
# translation = tanslate_sentence_list(sentences)
# for index, sentence in enumerate(sentences):
#     print(f'Sentence: {sentence}')
#     print(f"Original labels: {labels[index]}, output labels: '{translation[index]}'")
=======
def main():
    # tanslate_one_sentence("He is running like a superman! :skull:")
    # return

    sentences, labels = read_csv(FILE_PATH)
    translation = tanslate_sentence_list(sentences)
    for index, sentence in enumerate(sentences):
        print(f'Sentence: {sentence}')
        print(f"Original labels: {labels[index]}, output labels: '{translation[index]}'")


if __name__ == "__main__":
    main()
>>>>>>> aab2000ee99c041ae6a3db2c68971c908523089e:test/T5/T5_test.py
