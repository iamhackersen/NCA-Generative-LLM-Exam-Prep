import React, { useState } from 'react';
import { batches } from './data';
import { Batch, QuestionStatus } from './types';
import StartScreen from './components/StartScreen';
import QuizCard from './components/QuizCard';
import ScoreScreen from './components/ScoreScreen';
import { ChevronLeft, ChevronRight, Menu } from 'lucide-react';

const App: React.FC = () => {
  // --- State ---
  const [activeBatchId, setActiveBatchId] = useState<string | null>(null);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  // Map question ID -> selected Option Indices
  const [userAnswers, setUserAnswers] = useState<Record<number, number[]>>({});
  // Map question ID -> status
  const [questionStatus, setQuestionStatus] = useState<Record<number, QuestionStatus>>({});
  const [isQuizComplete, setIsQuizComplete] = useState(false);

  // --- Derived State ---
  const activeBatch = batches.find(b => b.id === activeBatchId);
  const currentQuestion = activeBatch ? activeBatch.questions[currentQuestionIndex] : null;

  // --- Handlers ---
  
  const handleStartBatch = (batchId: string) => {
    setActiveBatchId(batchId);
    setCurrentQuestionIndex(0);
    setUserAnswers({});
    setQuestionStatus({});
    setIsQuizComplete(false);
  };

  const handleReturnHome = () => {
    setActiveBatchId(null);
  };

  const handleRetry = () => {
    if (activeBatchId) handleStartBatch(activeBatchId);
  };

  const handleOptionSelect = (optionIndex: number) => {
    if (!currentQuestion) return;
    
    const qId = currentQuestion.id;
    const type = currentQuestion.type;
    const currentSelected = userAnswers[qId] || [];

    let newSelected: number[];

    if (type === 'single') {
        newSelected = [optionIndex];
    } else {
        if (currentSelected.includes(optionIndex)) {
            newSelected = currentSelected.filter(i => i !== optionIndex);
        } else {
            // Limit to 2 for simplicity if needed, but standard multi-select allows more usually. 
            // Provided questions say "Select TWO", but UI should allow standard toggle.
            newSelected = [...currentSelected, optionIndex];
        }
    }
    
    setUserAnswers(prev => ({ ...prev, [qId]: newSelected }));
  };

  const handleCheckAnswer = () => {
    if (!currentQuestion) return;
    const qId = currentQuestion.id;
    const selected = userAnswers[qId] || [];
    
    if (selected.length === 0) return;

    // Sort to compare arrays
    const sortedSelected = [...selected].sort();
    const sortedCorrect = [...currentQuestion.correct].sort();
    const isCorrect = JSON.stringify(sortedSelected) === JSON.stringify(sortedCorrect);

    setQuestionStatus(prev => ({ 
        ...prev, 
        [qId]: isCorrect ? 'correct' : 'wrong' 
    }));
  };

  const navigateQuestion = (delta: number) => {
    if (!activeBatch) return;
    const newIndex = currentQuestionIndex + delta;
    if (newIndex >= 0 && newIndex < activeBatch.questions.length) {
        setCurrentQuestionIndex(newIndex);
    } else if (newIndex >= activeBatch.questions.length) {
        finishQuiz();
    }
  };

  const finishQuiz = () => {
      setIsQuizComplete(true);
  };

  const calculateScore = () => {
      if (!activeBatch) return 0;
      let score = 0;
      activeBatch.questions.forEach(q => {
          if (questionStatus[q.id] === 'correct') score++;
      });
      return score;
  };

  // --- Render ---

  return (
    <div className="min-h-screen bg-nvidia-dark text-white font-sans flex flex-col">
      {/* Navbar */}
      <nav className="bg-black/50 backdrop-blur-md border-b border-white/10 sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
             <div className="w-8 h-8 bg-nvidia-green text-black font-bold flex items-center justify-center rounded">
                AI
             </div>
             <h1 className="font-semibold text-lg tracking-wide hidden sm:block">
                NVIDIA Certified Associate Generative LLM <span className="text-nvidia-gray font-normal">| Prep</span>
             </h1>
          </div>
          {activeBatch && !isQuizComplete && (
              <div className="flex items-center gap-4">
                  <div className="text-sm text-nvidia-gray">
                      Q{currentQuestionIndex + 1} / {activeBatch.questions.length}
                  </div>
                  <button onClick={handleReturnHome} className="p-2 hover:bg-white/10 rounded-full transition-colors" title="Exit Batch">
                      <Menu size={20} />
                  </button>
              </div>
          )}
        </div>
        
        {/* Progress Bar */}
        {activeBatch && !isQuizComplete && (
            <div className="h-1 bg-nvidia-panel w-full">
                <div 
                    className="h-full bg-nvidia-green transition-all duration-300 ease-out"
                    style={{ width: `${((currentQuestionIndex) / activeBatch.questions.length) * 100}%` }}
                />
            </div>
        )}
      </nav>

      {/* Main Content */}
      <main className="flex-grow flex flex-col items-center justify-start p-4 sm:p-6 md:p-8">
        
        {/* 1. Start Screen */}
        {!activeBatch && (
           <StartScreen batches={batches} onSelectBatch={handleStartBatch} />
        )}

        {/* 2. Quiz Interface */}
        {activeBatch && !isQuizComplete && currentQuestion && (
            <div className="w-full max-w-4xl flex flex-col h-full">
                <div className="flex-grow">
                    <QuizCard 
                        question={currentQuestion}
                        selectedOptions={userAnswers[currentQuestion.id] || []}
                        status={questionStatus[currentQuestion.id] || 'unanswered'}
                        onOptionSelect={handleOptionSelect}
                        onCheck={handleCheckAnswer}
                    />
                </div>
                
                {/* Navigation Footer */}
                <div className="mt-8 pt-6 border-t border-white/10 flex justify-between items-center max-w-3xl mx-auto w-full">
                    <button 
                        onClick={() => navigateQuestion(-1)}
                        disabled={currentQuestionIndex === 0}
                        className="flex items-center gap-2 text-nvidia-gray hover:text-white disabled:opacity-30 disabled:hover:text-nvidia-gray transition-colors"
                    >
                        <ChevronLeft size={20} /> Previous
                    </button>

                    <div className="flex gap-1">
                        {activeBatch.questions.map((q, idx) => {
                            let dotColor = "bg-nvidia-panel";
                            if (idx === currentQuestionIndex) dotColor = "bg-white scale-125";
                            else if (questionStatus[q.id] === 'correct') dotColor = "bg-nvidia-green";
                            else if (questionStatus[q.id] === 'wrong') dotColor = "bg-red-500";
                            
                            return (
                                <div key={idx} className={`w-2 h-2 rounded-full transition-all ${dotColor}`} />
                            )
                        })}
                    </div>

                    <button 
                        onClick={() => navigateQuestion(1)}
                        className="flex items-center gap-2 text-white hover:text-nvidia-green font-semibold transition-colors"
                    >
                        {currentQuestionIndex === activeBatch.questions.length - 1 ? 'Finish' : 'Next'} <ChevronRight size={20} />
                    </button>
                </div>
            </div>
        )}

        {/* 3. Score Screen */}
        {activeBatch && isQuizComplete && (
            <ScoreScreen 
                score={calculateScore()} 
                total={activeBatch.questions.length}
                onRetry={handleRetry}
                onHome={handleReturnHome}
            />
        )}

      </main>
    </div>
  );
};

export default App;