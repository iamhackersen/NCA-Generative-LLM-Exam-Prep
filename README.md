# NVIDIA Certified Associate Generative LLM | Prep

A comprehensive, web-based practice interface designed to help candidates prepare for the **NVIDIA Certified Associate - Generative AI LLM** exam. This application consolidates multiple practice question batches into a single, responsive, and interactive React application.

test-page https://nca-generative-llm-exam-prep.vercel.app/
<img width="1184" height="777" alt="main layout" src="https://github.com/user-attachments/assets/a2f8ff16-1831-42e7-ac1c-36e2f71eb61a" />
<img width="1150" height="830" alt="selection" src="https://github.com/user-attachments/assets/c67f77c1-9bd1-420d-94b1-f09fc2e6fe47" />
<img width="1113" height="888" alt="check answer" src="https://github.com/user-attachments/assets/39441347-fb9c-4db7-8caa-ff75e0519142" />
<img width="1162" height="896" alt="incorrect answer" src="https://github.com/user-attachments/assets/86ad56e5-4ea6-4230-ac67-2b3036962e15" />

## 🚀 Features

- **Multiple Practice Modules**: Organize questions into distinct batches (e.g., Core ML, Software Dev, Experimentation).
- **Interactive Quiz Interface**: 
  - Select single or multiple answers.
  - Immediate "Check Answer" feedback.
  - Detailed explanations for every question to aid learning.
  - Visual distinction between correct, incorrect, and selected states.
- **Progress Tracking**: Real-time progress bar and question navigation.
- **Score Summary**: Visual circular progress ring and performance assessment upon completion.
- **Responsive Design**: Fully optimized for desktop, tablet, and mobile devices using Tailwind CSS.
- **Dark Mode UI**: Clean, minimalist aesthetic inspired by NVIDIA's brand identity.

## 🛠️ Tech Stack

- **Framework**: [React](https://react.dev/) (v19)
- **Language**: [TypeScript](https://www.typescriptlang.org/)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/)
- **Icons**: [Lucide React](https://lucide.dev/)
- **Build Tool**: Vite (Recommended for local development)

## Prompt Engineering
#Content
Create sample questions mimic the real exam questions for Nvidia Certified Associate Generative LLM
Core Machine Learning and AI Knowledge 6 questions, Software Development 5 questions Experimentation 4 questions, Data
Analysis and Visualization 3 questions and Trustworthy AI 2 questions
Label the questions with topic likes Core Machine Learning and AI Knowledge , Software Development, Experimentation, Data
Analysis and Visualization and Trustworthy AI
Contain 2 type of question, one with single answer, another with multiple choices
Questions shall not be repetitive
Selection of answers limit to 4, for multiple choice select any 2 questions
Randomly assigned the answers, i.e. shall not be always B or As or A and C
Create first 20 questions 15 for single answer, 5 for multiple choices for preview
Provide 50 words of answer explanation

#Interface and UI design
Create a html format
Minimalist / compact design fit onto one page without need to scroll up or down
Use color green and dark mode
One page one question with 4 answer selection
Include Previous / Next question button
Selection of answer shall have blue frame on the edge box, same for multiple choice.
Can reselect the answer by clicking on new selection.
Create the answer button at the button of every question, do not reveal the answer automatically
If the answer is incorrect, do not reveal the answer, but allow user to reselect the answer, once user click on check answer button, ser
can not reselect the answer, same to the score. once the check answer is clicked, the score must be calculated.
previous button on left center, check answer on center, next button on center right
Include enough space to show the 50words explanation
Include completion status bar
Create a html format
Add total score onto the final question. I.e. xx/20 xx below 15 (Study harder!) 15/20 (Keep trying!) 18/20 (Very Good!) or 20/20
(Outstanding!)


## 📂 Project Structure

```
├── components/
│   ├── QuizCard.tsx       # Renders individual questions and options
│   ├── ScoreScreen.tsx    # Displays final results and navigation
│   └── StartScreen.tsx    # Main menu to select practice batches
├── App.tsx                # Main application logic and state management
├── data.ts                # Contains all question data (The "Database")
├── index.html             # Entry HTML with Tailwind CDN
├── index.tsx              # React entry point
└── types.ts               # TypeScript interfaces for Questions and Batches
```

## ⚡ Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn

### Installation

1.  **Clone the repository** (or download the source files):
    ```bash
    git clone https://github.com/your-username/nvidia-genai-prep.git
    cd nvidia-genai-prep
    ```

2.  **Install dependencies**:
    ```bash
    npm install
    ```

3.  **Run the development server**:
    ```bash
    npm run dev
    ```

4.  **Open in Browser**:
    Navigate to `http://localhost:5173` (or the port shown in your terminal).

## 📝 How to Add Questions

All question data is stored in `data.ts`. To add new questions or create a new batch:

1.  Open `data.ts`.
2.  Define a new array of questions following the `Question` interface:
    ```typescript
    const myNewBatch = [
      {
          id: 101,
          topic: "New Topic",
          type: "single", // or "multiple"
          text: "Your question text here?",
          options: ["Option A", "Option B", "Option C", "Option D"],
          correct: [0], // Index of the correct answer (0-based)
          explanation: "Detailed explanation here."
      },
      // ... more questions
    ];
    ```
3.  Add the new batch to the `batches` array at the bottom of the file:
    ```typescript
    export const batches: Batch[] = [
        // ... existing batches
        { id: 'b5', name: 'New Practice Module', questions: myNewBatch },
    ];
    ```

## 🎨 Customization

- **Colors**: The theme is defined in the `tailwind.config` script within `index.html`. You can modify the `nvidia` color palette to change the branding.
- **Icons**: The app uses `lucide-react`. You can easily swap icons in the component files.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

*Disclaimer: This application is a community-created study tool and is not officially affiliated with or endorsed by NVIDIA Corporation.*
