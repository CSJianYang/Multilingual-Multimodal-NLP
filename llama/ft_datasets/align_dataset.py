import numpy as np 

class AlignDataset:
    def __init__(self, subword_align_dataset, phrase_align_dataset=None):
        self.subword_align_question = []
        self.subword_align_answer   = []
        self.phrase_align_question  = []
        self.phrase_align_answer    = []


        for i in range(len(subword_align_dataset)//2):
            self.subword_align_question.append(subword_align_dataset[2*i])
            self.subword_align_answer.append(subword_align_dataset[2*i+1])

            self.phrase_align_question.append(phrase_align_dataset[2*i])
            self.phrase_align_answer.append(phrase_align_dataset[2*i+1])

        assert len(self.subword_align_answer) == len(self.subword_align_question)
        assert len(self.phrase_align_answer)  == len(self.phrase_align_question)

    # def reversed_align_dataset(self):
    #     if self.subword_align_dataset is not None:
    #         reversed_subword_align_dataset = []
    #         for i in range(len(self.subword_align_dataset)):
    #             reversed_subword_align_dataset.append(np.flip(self.subword_align_dataset[i], 1) if len(self.subword_align_dataset[i]) > 0 else np.array([]))
    #     else:
    #         reversed_subword_align_dataset = None

    #     if self.phrase_align_dataset is not None:
    #         reversed_phrase_align_dataset = []
    #         for i in range(len(self.phrase_align_dataset)):
    #             reversed_phrase_align_dataset.append(np.flip(self.phrase_align_dataset[i], 1) if len(self.phrase_align_dataset[i]) > 0 else np.array([]))
    #     else:
    #         reversed_phrase_align_dataset = None
    #     return reversed_subword_align_dataset, reversed_phrase_align_dataset

    def __getitem__(self, index):
        return (self.subword_align_question[index],self.subword_align_answer[index])  if self.subword_align_question is not None else None, (self.phrase_align_question[index], self.phrase_align_answer[index]) if self.phrase_align_question is not None else None

    def __len__(self):
        if self.subword_align_question is not None:
            return len(self.subword_align_question)
        else:
            return len(self.phrase_align_question)