export type QuestionType = 'single' | 'multiple';

export interface Question {
  id: number;
  topic: string;
  type: QuestionType;
  text: string;
  options: string[];
  correct: number[];
  explanation: string;
}

export interface Batch {
  id: string;
  name: string;
  questions: Question[];
}

export type QuestionStatus = 'unanswered' | 'correct' | 'wrong';

export interface QuizState {
  currentBatchId: string | null;
  currentQuestionIndex: number;
  userAnswers: Record<number, number[]>; // Map question index to selected option indices
  questionStatus: Record<number, QuestionStatus>; // Map question index to status
  score: number;
  isComplete: boolean;
}