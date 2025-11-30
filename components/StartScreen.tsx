import React from 'react';
import { Batch } from '../types';
import { PlayCircle, Database, Layout } from 'lucide-react';

interface StartScreenProps {
  batches: Batch[];
  onSelectBatch: (batchId: string) => void;
}

const StartScreen: React.FC<StartScreenProps> = ({ batches, onSelectBatch }) => {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-8 animate-fade-in">
      <div className="text-center space-y-4">
        <div className="bg-nvidia-green/10 text-nvidia-green w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 border border-nvidia-green/30">
          <Database size={32} />
        </div>
        <h2 className="text-3xl font-bold text-white tracking-tight">Select Practice Module</h2>
        <p className="text-nvidia-gray max-w-md mx-auto">
          Choose a question batch to begin your preparation for the NVIDIA Certified Associate Generative LLM exam.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-2xl">
        {batches.map((batch) => (
          <button
            key={batch.id}
            onClick={() => onSelectBatch(batch.id)}
            className="group relative flex items-center p-6 bg-nvidia-panel border border-transparent rounded-lg hover:border-nvidia-green transition-all duration-300 hover:shadow-[0_0_15px_rgba(118,185,0,0.2)] text-left"
          >
            <div className="mr-4 bg-black/30 p-3 rounded-md group-hover:bg-nvidia-green group-hover:text-black transition-colors duration-300">
               <Layout size={24} />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-white group-hover:text-nvidia-green transition-colors">
                {batch.name}
              </h3>
              <p className="text-sm text-nvidia-gray mt-1">
                {batch.questions.length} Questions
              </p>
            </div>
            <div className="absolute right-4 opacity-0 group-hover:opacity-100 transition-opacity transform translate-x-2 group-hover:translate-x-0">
                <PlayCircle className="text-nvidia-green" size={24} />
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

export default StartScreen;