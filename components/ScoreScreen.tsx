import React from 'react';
import { Award, RotateCcw, Home } from 'lucide-react';

interface ScoreScreenProps {
  score: number;
  total: number;
  onRetry: () => void;
  onHome: () => void;
}

const ScoreScreen: React.FC<ScoreScreenProps> = ({ score, total, onRetry, onHome }) => {
  const percentage = Math.round((score / total) * 100);
  
  let message = "Study harder! Review the core concepts.";
  if (percentage === 100) message = "Outstanding! You are a Master.";
  else if (percentage >= 80) message = "Very Good! Excellent performance.";
  else if (percentage >= 60) message = "Keep trying! Solid foundation.";

  const strokeDasharray = 2 * Math.PI * 54; // radius ~54
  const strokeDashoffset = strokeDasharray * ((100 - percentage) / 100);

  return (
    <div className="flex flex-col items-center justify-center h-full space-y-8 animate-fade-in py-10">
      <Award size={64} className="text-nvidia-green mb-4" />
      
      <h2 className="text-3xl font-bold text-white">Assessment Complete</h2>
      
      {/* Circular Progress */}
      <div className="relative w-48 h-48">
         <svg className="w-full h-full transform -rotate-90">
             <circle 
                cx="96" cy="96" r="54" 
                stroke="#2d2d2d" strokeWidth="12" fill="transparent" 
             />
             <circle 
                cx="96" cy="96" r="54" 
                stroke="#76b900" strokeWidth="12" fill="transparent"
                strokeDasharray={strokeDasharray}
                strokeDashoffset={strokeDashoffset}
                strokeLinecap="round"
                className="transition-all duration-1000 ease-out"
             />
         </svg>
         <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-4xl font-bold text-white">{score}</span>
            <span className="text-sm text-nvidia-gray">out of {total}</span>
         </div>
      </div>

      <div className="text-center space-y-2">
          <p className="text-xl text-nvidia-green font-semibold">{percentage}%</p>
          <p className="text-nvidia-text">{message}</p>
      </div>

      <div className="flex gap-4">
          <button 
            onClick={onHome}
            className="flex items-center gap-2 px-6 py-3 bg-nvidia-panel hover:bg-nvidia-hover text-white rounded transition-colors"
          >
             <Home size={18} /> Home
          </button>
          <button 
            onClick={onRetry}
            className="flex items-center gap-2 px-6 py-3 bg-nvidia-green hover:bg-[#6aa600] text-black font-bold rounded transition-colors"
          >
             <RotateCcw size={18} /> Retry Batch
          </button>
      </div>
    </div>
  );
};

export default ScoreScreen;