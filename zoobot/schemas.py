import logging

from typing import List

class Question():

    def __init__(self, question_text:str, answer_text: List, label_cols:List):
        self.text = question_text
        self.answers = create_answers(self, answer_text, label_cols)  # passing a reference to self, will stay up-to-date

        self.start_index = min(a.index for a in self.answers)
        self.end_index = max(a.index for a in self.answers)
        assert [self.start_index <= a.index <= self.end_index for a in self.answers]

        self._asked_after = None

    @property
    def asked_after(self):
        return self._asked_after

    def __repr__(self):
        return f'{self.text}, indices {self.start_index} to {self.end_index}, asked after {self.asked_after}'


class Answer():

    def __init__(self, text, question, index):
        self.text = text
        self.question = question

        self.index = index
        self._next_question = None

    @property
    def next_question(self):
        return self._next_question

    def __repr__(self):
        return f'{self.text}, index {self.index}'

    @property
    def pretty_text(self):
        return self.text.replace('-',' ').replace('_', ' ').title()


def create_answers(question:Question, answers_texts:List, label_cols:List):
    question_text = question.text
    answers = []
    for answer_text in answers_texts:
        answers.append(
            Answer(
                text=question_text + answer_text,  # e.g. smooth-or-featured_smooth
                question=question,
                index=label_cols.index(question_text + answer_text)  # will hopefully throw value error if not found?
                # _next_question not set, set later with dependancies
            )
        )
    return answers
    

def set_dependencies(questions, dependencies):

    for question in questions:
        prev_answer_text = dependencies[question.text]
        if prev_answer_text is not None:
            prev_answer = [a for q in questions for a in q.answers if a.text == prev_answer_text][0]  # will be exactly one match
            prev_answer._next_question = question
            question._asked_after = prev_answer
    # acts inplace


class Schema():
    """
Relate the df label columns tor question/answer groups and to tfrecod label indices
    """
    def __init__(self, question_answer_pairs:dict, dependencies):
        """
        Requires that labels be continguous by question - easily satisfied
        
        Args:
            question_answer_pairs (Dict): e.g. {'smooth-or-featured: ['_smooth, _featured-or-disk, ...], ...}
        """
        logging.info(f'Q/A pairs: {question_answer_pairs}')
        self.question_answer_pairs = question_answer_pairs
        _, self.label_cols = extract_questions_and_label_cols(question_answer_pairs)
        self.dependencies = dependencies
        """
        Be careful:
        - first entry should be the first answer to that question, by df column order
        - second entry should be the last answer to that question, similarly
        - answers in between will be included: these are used to slice
        - df columns must be contigious by question (e.g. not smooth_yes, bar_no, smooth_no) for this to work!
        """
        self.questions = [Question(question_text, answers_text, self.label_cols) for question_text, answers_text in question_answer_pairs.items()]
        if len(self.questions) > 1:
            set_dependencies(self.questions, self.dependencies)

        assert len(self.question_index_groups) > 0
        assert len(self.questions) == len(self.question_index_groups)

        print(self.named_index_groups)

    def get_answer(self, answer_text):
        try:
            return [a for q in self.questions for a in q.answers if a.text == answer_text][0]  # will be exactly one match
        except IndexError:
            raise ValueError('Answer not found: ', answer_text)

    def get_question(self, question_text):
        try:
            return [q for q in self.questions if q.text == question_text][0]  # will be exactly one match
        except  IndexError:
            raise ValueError('Question not found: ', question_text)
    
    @property
    def question_index_groups(self):
         # start and end indices of answers to each question in label_cols e.g. [[0, 1]. [1, 3]] 
        return [(q.start_index, q.end_index) for q in self.questions]


    @property
    def named_index_groups(self):
        return dict(zip(self.questions, self.question_index_groups))


    def joint_p(self, prob_of_answers, answer_text):
        assert prob_of_answers.ndim == 2  # batch, p. No 'per model', marginalise first
        # prob(answer) = p(that answer|that q asked) * p(that q_asked) i.e...
        # prob(answer) = p(that answer|that q asked) * p(answer before that q)
        answer = self.get_answer(answer_text)
        p_answer_given_question = prob_of_answers[:, answer.index]

        question = answer.question
        prev_answer = question.asked_after
        if prev_answer is None:
            return p_answer_given_question
        else:
            p_prev_answer = self.joint_p(prob_of_answers, prev_answer.text)  # recursive
            return p_answer_given_question * p_prev_answer

    @property
    def answers(self):
        answers = []
        for q in self.questions:
            for a in q.answers:
                answers.append(a)
        return answers

    # TODO write to disk


def extract_questions_and_label_cols(question_answer_pairs):
    questions = list(question_answer_pairs.keys())
    label_cols = [q + answer for q, answers in question_answer_pairs.items() for answer in answers]
    return questions, label_cols
