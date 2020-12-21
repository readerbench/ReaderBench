from rb.processings.summarization.text_rank import get_sentence_scores
from rb.core.lang import Lang
from rb.core.document import Document
from operator import itemgetter

if __name__ == "__main__":
   with open("texts.txt", "rt") as f:
      for line in f:
         doc = Document(Lang.EN, line)
         scores = get_sentence_scores(doc)
         ranking = [0] * len(scores)
         for i, score in enumerate(scores):
            ranking[score[0]] = i
         print(ranking)

    # dataset_parser = DUC2001Parser(Path.cwd() / "corpus" / "DUC2001")
    # summarize_duc2001_dataset(spacy_parser, dataset_parser, summarizer)

    # summarize_dataset(dataset_type=DatasetType.DUC_2002, summarizer=submodular_summarizer)

