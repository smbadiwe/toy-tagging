import torch
import numpy as np
import torch
import torch.optim as optim
from allennlp.common.file_utils import cached_path
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.training.trainer import Trainer

from dataset_readers.posdatareader import PosDatasetReader
from models.lstm_tagger import LstmTagger

torch.manual_seed(1)


def run():
    reader = PosDatasetReader()
    train_dataset = reader.read(cached_path(
        'https://raw.githubusercontent.com/allenai/allennlp'
        '/master/tutorials/tagger/training.txt'))
    validation_dataset = reader.read(cached_path(
        'https://raw.githubusercontent.com/allenai/allennlp'
        '/master/tutorials/tagger/validation.txt'))
    vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6
    if torch.cuda.is_available():
        cuda_device = 0
    else:
        cuda_device = -1

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

    def train_and_save():
        model = LstmTagger(word_embeddings, lstm, vocab)
        if cuda_device >= 0:
            model = model.cuda(cuda_device)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])
        iterator.index_with(vocab)
        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          iterator=iterator,
                          train_dataset=train_dataset,
                          validation_dataset=validation_dataset,
                          patience=10,
                          num_epochs=500,
                          cuda_device=cuda_device)
        trainer.train()
        predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
        tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
        tag_ids = np.argmax(tag_logits, axis=-1)
        print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])
        # Here's how to save the model.
        with open("./tmp/model.th", 'wb') as f:
            torch.save(model.state_dict(), f)
        vocab.save_to_files("./tmp/vocabulary")
        return tag_logits

    def reload_and_test(tag_logits):
        # And here's how to reload the model.
        vocab2 = Vocabulary.from_files("./tmp/vocabulary")
        model2 = LstmTagger(word_embeddings, lstm, vocab2)
        with open("./tmp/model.th", 'rb') as f:
            model2.load_state_dict(torch.load(f))
        if cuda_device > -1:
            model2.cuda(cuda_device)
        predictor2 = SentenceTaggerPredictor(model2, dataset_reader=reader)
        tag_logits2 = predictor2.predict("The dog ate the apple")['tag_logits']
        np.testing.assert_array_almost_equal(tag_logits2, tag_logits)
        tag_logits2 = predictor2.predict("The cat killed the rat")['tag_logits']
        tag_ids = np.argmax(tag_logits2, axis=-1)
        print([model2.vocab.get_token_from_index(i, 'labels') for i in tag_ids])

    _tag_logits = train_and_save()
    reload_and_test(_tag_logits)


if __name__ == "__main__":
    run()
