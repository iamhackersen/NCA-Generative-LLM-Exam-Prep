# NVIDIA Certified Associate Generative LLM | Prep

A comprehensive, web-based practice interface designed to help candidates prepare for the **NVIDIA Certified Associate - Generative AI LLM** exam. This application consolidates multiple practice question batches into a single, responsive, and interactive React application.

![App Screenshot Placeholder](https://via.placeholder.com/800x400?text=Application+Screenshot)

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