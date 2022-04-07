import logging
from typing import List

import numpy as np

from zoobot.shared import label_metadata


class Question():

    def __init__(self, question_text:str, answer_text: List, label_cols:List):
        """
        Class representing decision tree question.
        Requires ``label_cols`` as an input in order to find the index (vs. all questions and answers) of this question and each answer.

        Args:
            question_text (str): e.g. 'smooth-or-featured'
            answer_text (List): e.g. ['smooth', 'featured-or-disk]
            label_cols (List): list of all questions and answers e.g. ['smooth-or-featured_smooth', 'smooth-or-featured_featured-or-disk', ...]
        """
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
        """
        Class representing decision tree answer.

        Each answer includes the answer text (often used as a column header),
        the corresponding question, and its index in ``label_cols`` (often used for slicing model outputs)

        Args:
            text (str): e.g. 'smooth-or-featured_smooth'
            question (Question): Question class for this answer
            index (int): index of answer in label_cols (0-33 for DECaLS)
        """
        self.text = text
        self.question = question

        self.index = index
        self._next_question = None

    @property
    def next_question(self):
        """
        
        Returns:
            Question: question that follows after this answer. None initially, added by ``set_dependancies``.
        """
        return self._next_question

    def __repr__(self):
        return f'{self.text}, index {self.index}'

    @property
    def pretty_text(self):
        """
        Returns:
            str: Nicely formatted text for plots etc
        """
        return self.text.replace('-',' ').replace('_', ' ').title()


def create_answers(question:Question, answers_texts:List, label_cols:List):
    """
    Instantiate the Answer classes for a given ``Question``.
    Each answer includes the answer text (often used as a column header),
    the corresponding question, and its index in ``label_cols`` (often used for slicing model outputs)

    Args:
        question (Question): question to which to create answers e.g. Question(smooth-or-featured)
        answers_texts (List): answer strings e.g. ['smooth-or-featured_smooth', 'smooth-or-featured_featured-or-disk']
        label_cols (List): list of all questions and answers e.g. ['smooth-or-featured_smooth', 'smooth-or-featured_featured-or-disk', ...]

    Returns:
        List: of Answers to that question e.g. [Answer(smooth-or-featured_smooth), Answer(smooth-or-featured_featured-or-disk)]
    """
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
    """
    Link each answer to question which follows, and vica versa.
    Acts inplace.

    Specifically, for every answer in every question, set answer._next question to refer to the Question which follows that answer.
    Then for that Question, set question._asked_after to that answer.

    Args:
        questions (List): of questions e.g. [Question('smooth-or-featured'), Question('edge-on-disk')]
        dependencies (dict): dict mapping each question (e.g. disk-edge-on) to the answer on which it depends (e.g. smooth-or-featured_featured-or-disk)
    """

    for question in questions:
        prev_answer_text = dependencies[question.text]
        if prev_answer_text is not None:
            try:
                prev_answer = [a for q in questions for a in q.answers if a.text == prev_answer_text][0]  # look through every answer, find those with the same text as "prev answer text" - will be exactly one match
            except IndexError:
                raise ValueError(f'{prev_answer_text} not found in dependencies')
            prev_answer._next_question = question
            question._asked_after = prev_answer


class Schema():
    def __init__(self, question_answer_pairs:dict, dependencies):
        """
        Relate the df label columns tor question/answer groups and to tfrecod label indices
        Requires that labels be continguous by question - easily satisfied
        
        Args:
            question_answer_pairs (dict): e.g. {'smooth-or-featured: ['_smooth, _featured-or-disk, ...], ...}
            dependencies (dict): dict mapping each question (e.g. disk-edge-on) to the answer on which it depends (e.g. smooth-or-featured_featured-or-disk)
        """
        logging.debug(f'Q/A pairs: {question_answer_pairs}')
        self.question_answer_pairs = question_answer_pairs
        _, self.label_cols = label_metadata.extract_questions_and_label_cols(question_answer_pairs)
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


    def get_answer(self, answer_text):
        """

        Args:
            answer_text (str): e.g. 'smooth-or-featured_smooth'

        Raises:
            ValueError: No answer with that answer_text found

        Returns:
            Answer: the answer with matching answer_text e.g. Answer('smooth-or-featured_smooth')
        """
        try:
            return [a for q in self.questions for a in q.answers if a.text == answer_text][0]  # will be exactly one match
        except IndexError:
            raise ValueError('Answer not found: ', answer_text)


    def get_question(self, question_text):
        """

        Args:
            question_text (str): e.g. 'smooth-or-featured'

        Raises:
            ValueError: No question with that question_text found

        Returns:
            Question: the question with matching question_text e.g. Question('smooth-or-featured')
        """
        try:
            return [q for q in self.questions if q.text == question_text][0]  # will be exactly one match
        except  IndexError:
            raise ValueError('Question not found: ', question_text)
    

    @property
    def question_index_groups(self):
        """

        Returns:
            Paired (tuple) integers of (first, last) indices of answers to each question, listed for all questions.
            Useful for slicing model predictions by question.
        """
         # start and end indices of answers to each question in label_cols e.g. [[0, 1]. [1, 3]] 
        return [(q.start_index, q.end_index) for q in self.questions]


    @property
    def named_index_groups(self):
        """
        
        Returns:
            dict: mapping each question to the start and end index of its answers in label_cols, e.g. {Question('smooth-or-featured'): [0, 2], ...}
        """
        return dict(zip(self.questions, self.question_index_groups))


    def joint_p(self, prob_of_answers, answer_text):
        """
        Probability of the answer with ``answer_text`` being asked, given the (predicted) probability of every answer.
        Useful for estimating the relevance of an answer e.g. to ignore predictons for answers less than 50% likely to be asked.

        Broadcasts over batch dimension.

        Args:
            prob_of_answers (np.ndarray): prob. of each answer being asked, of shape (galaxies, answers) where the index of answers matches label_cols
            answer_text (str): which answer to find the prob. of being asked e.g. 'edge-on-disk_yes'

        Returns:
            np.ndarray: prob of that answer being asked e.g. 0.5 for 'edge-on-disk_yes' of prob of 'smooth-or-featured_featured-or-disk' is 0.5. Shape (galaxies).
        """
        assert prob_of_answers.ndim == 2  # batch, p. No 'per model', marginalise first
        # prob(answer) = p(that answer|that q asked) * p(that q_asked) i.e...
        # prob(answer) = p(that answer|that q asked) * p(answer before that q)
        answer = self.get_answer(answer_text)
        p_answer_given_question = prob_of_answers[:, answer.index]
        if all(np.isnan(p_answer_given_question)):
            logging.warning(f'All p_answer_given_question for {answer_text} ({answer.index}) are nan i.e. all fractions are nan - check that labels for this question are appropriate')

        question = answer.question
        prev_answer = question.asked_after
        if prev_answer is None:
            return p_answer_given_question
        else:
            p_prev_answer = self.joint_p(prob_of_answers, prev_answer.text)  # recursive
            return p_answer_given_question * p_prev_answer


    @property
    def answers(self):
        """

        Returns:
            list: all answers
        """
        answers = []
        for q in self.questions:
            for a in q.answers:
                answers.append(a)
        return answers
