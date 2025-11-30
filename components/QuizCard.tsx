import React from 'react';
import { Question, QuestionStatus } from '../types';
import { CheckCircle, XCircle, AlertCircle } from 'lucide-react';

interface QuizCardProps {
  question: Question;
  selectedOptions: number[];
  status: QuestionStatus;
  onOptionSelect: (optionIndex: number) => void;
  onCheck: () => void;
}

const QuizCard: React.FC<QuizCardProps> = ({ 
  question, 
  selectedOptions, 
  status, 
  onOptionSelect, 
  onCheck 
}) => {
  
  const isAnswered = status !== 'unanswered';
  const isMultiple = question.type === 'multiple';

  return (
    <div className="w-full max-w-3xl mx-auto space-y-6">
      {/* Header Info */}
      <div className="flex justify-between items-center mb-2">
         <span className="bg-nvidia-panel border border-nvidia-green/30 text-nvidia-green px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider">
            {question.topic}
         </span>
         {isMultiple && (
             <span className="text-xs text-nvidia-gray flex items-center gap-1">
                <AlertCircle size={12} /> Select Two
             </span>
         )}
      </div>

      {/* Question Text */}
      <h2 className="text-xl md:text-2xl font-light leading-relaxed text-white">
        {question.text}
      </h2>

      {/* Options */}
      <div className="space-y-3">
        {question.options.map((option, idx) => {
          const isSelected = selectedOptions.includes(idx);
          const isCorrect = question.correct.includes(idx);
          
          let cardStyle = "border-nvidia-panel hover:bg-nvidia-hover"; // Default
          
          if (isAnswered) {
             if (isCorrect) {
                cardStyle = "border-nvidia-green bg-nvidia-green/10";
             } else if (isSelected && !isCorrect) {
                cardStyle = "border-red-500 bg-red-500/10 opacity-70";
             } else {
                cardStyle = "border-transparent opacity-50";
             }
          } else if (isSelected) {
             cardStyle = "border-blue-500 bg-nvidia-panel";
          }

          return (
            <div
              key={idx}
              onClick={() => !isAnswered && onOptionSelect(idx)}
              className={`
                relative p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 flex items-center
                ${cardStyle}
                ${!isAnswered ? 'active:scale-[0.99]' : 'cursor-default'}
              `}
            >
              <div className={`
                flex-shrink-0 w-8 h-8 rounded flex items-center justify-center mr-4 text-sm font-bold border
                ${isSelected || (isAnswered && isCorrect) ? 'bg-nvidia-panel border-transparent' : 'border-nvidia-gray/30 text-nvidia-gray'}
              `}>
                {String.fromCharCode(65 + idx)}
              </div>
              <span className="text-nvidia-text text-sm md:text-base">{option}</span>
              
              {isAnswered && isCorrect && (
                  <CheckCircle className="absolute right-4 text-nvidia-green" size={20} />
              )}
              {isAnswered && isSelected && !isCorrect && (
                  <XCircle className="absolute right-4 text-red-500" size={20} />
              )}
            </div>
          );
        })}
      </div>

      {/* Explanation */}
      {isAnswered && (
        <div className={`
            mt-6 p-5 rounded-lg border-l-4 animate-slide-up
            ${status === 'correct' ? 'bg-nvidia-green/10 border-nvidia-green' : 'bg-red-500/10 border-red-500'}
        `}>
          <div className="flex items-center gap-2 mb-2 font-bold">
            {status === 'correct' ? (
                <span className="text-nvidia-green">✓ Correct</span>
            ) : (
                <span className="text-red-500">✗ Incorrect</span>
            )}
          </div>
          <p className="text-sm md:text-base text-gray-300 leading-relaxed">
            {question.explanation}
          </p>
        </div>
      )}

      {/* Actions */}
      {!isAnswered && (
          <div className="flex justify-end pt-4">
              <button
                onClick={onCheck}
                disabled={selectedOptions.length === 0}
                className="bg-nvidia-green hover:bg-[#6aa600] disabled:opacity-50 disabled:cursor-not-allowed text-black font-bold py-3 px-8 rounded transition-colors duration-200 uppercase text-sm tracking-wide"
              >
                Check Answer
              </button>
          </div>
      )}
    </div>
  );
};

export default QuizCard;