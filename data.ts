import { Batch } from './types';

// Batch 1 Data
const batch1Questions = [
    {
        id: 1,
        topic: "Core Machine Learning",
        type: "single" as const,
        text: "Which regularization technique is primarily used to prevent overfitting by randomly deactivating neurons during the training process?",
        options: ["Batch Normalization", "Dropout", "Gradient Clipping", "Weight Decay"],
        correct: [1],
        explanation: "Dropout is a regularization technique that randomly deactivates neurons during training. This forces the network to learn more robust features and prevents over-reliance on specific neurons."
    },
    {
        id: 2,
        topic: "Core Machine Learning",
        type: "single" as const,
        text: "In the context of Large Language Models (LLMs), what is the primary role of the 'Attention Mechanism' within the Transformer architecture?",
        options: ["To compress image data", "To assign importance weights to different words in a sequence", "To convert text into audio waveforms", "To reduce the dimensionality of the dataset"],
        correct: [1],
        explanation: "The Attention Mechanism allows the model to weigh the relevance of different words in the input sequence when generating an output, solving the long-range dependency problem."
    },
    {
        id: 3,
        topic: "Core Machine Learning",
        type: "multiple" as const,
        text: "Select the TWO learning paradigms that are most commonly associated with training Generative Adversarial Networks (GANs).",
        options: ["Supervised Learning (Generator)", "Unsupervised Learning (Discriminator)", "Generative Modeling", "Discriminative Modeling"],
        correct: [2, 3], 
        explanation: "GANs consist of a Generator (Generative Modeling) and a Discriminator (Discriminative Modeling). They compete to map random noise to a data distribution."
    },
    {
        id: 4,
        topic: "Core Machine Learning",
        type: "single" as const,
        text: "What is 'Tokenization' in the pipeline of processing text for a Generative AI model?",
        options: ["Converting text into numerical vectors", "Breaking text into smaller units (words or sub-words)", "Removing stop words only", "Translating text to another language"],
        correct: [1],
        explanation: "Tokenization is the process of breaking down raw text into smaller units called tokens, which are subsequently converted into numerical embeddings."
    },
    {
        id: 5,
        topic: "Core Machine Learning",
        type: "single" as const,
        text: "Which optimization algorithm is most fundamental to training neural networks by iteratively adjusting weights to minimize the loss function?",
        options: ["K-Means Clustering", "Gradient Descent", "Principal Component Analysis", "Random Forest"],
        correct: [1],
        explanation: "Gradient Descent calculates the gradient of the error with respect to parameters and updates weights to find the local minimum."
    },
    {
        id: 6,
        topic: "Core Machine Learning",
        type: "single" as const,
        text: "What is an 'Embedding' in the context of Natural Language Processing?",
        options: ["A dense vector representation of data where similar items are close in space", "A method to compress files", "A rule-based grammar system", "A database of labeled images"],
        correct: [0],
        explanation: "Embeddings are low-dimensional, continuous vector representations where words with similar meanings have vectors that are geometrically close."
    },
    {
        id: 7,
        topic: "Software Development",
        type: "single" as const,
        text: "Which Python library is the industry standard for manipulating and analyzing structured data using DataFrames?",
        options: ["NumPy", "Pandas", "Matplotlib", "PyTorch"],
        correct: [1],
        explanation: "Pandas provides the DataFrame structure, allowing for easy handling of tabular data essential for AI data preparation."
    },
    {
        id: 8,
        topic: "Software Development",
        type: "multiple" as const,
        text: "When deploying a Generative AI model, which TWO technologies are commonly used to containerize the application for consistent deployment?",
        options: ["Docker", "Kubernetes", "Jupyter Notebook", "Git"],
        correct: [0, 1],
        explanation: "Docker creates containers for the application; Kubernetes orchestrates, scales, and manages these containers."
    },
    {
        id: 9,
        topic: "Software Development",
        type: "single" as const,
        text: "What is the primary purpose of NVIDIA NGC (NVIDIA GPU Cloud)?",
        options: ["A social network for developers", "A hub for GPU-optimized AI software, models, and containers", "A cloud storage for personal photos", "A programming language for GPUs"],
        correct: [1],
        explanation: "NVIDIA NGC is a catalog of GPU-optimized software, pre-trained models, and SDKs optimized to run efficiently on NVIDIA GPUs."
    },
    {
        id: 10,
        topic: "Software Development",
        type: "single" as const,
        text: "Which architectural style is commonly used to expose a Generative AI model as a web service?",
        options: ["Monolithic Architecture", "REST API", "Peer-to-Peer", "Mainframe"],
        correct: [1],
        explanation: "REST APIs allow client applications to send input data via HTTP requests and receive the model's output."
    },
    {
        id: 11,
        topic: "Software Development",
        type: "single" as const,
        text: "In the context of collaborative AI development, what is 'Git' primarily used for?",
        options: ["Training models", "Version control for code and tracking changes", "Visualizing data", "Cleaning datasets"],
        correct: [1],
        explanation: "Git is a version control system that tracks changes in source code and manages branches for collaborative development."
    },
    {
        id: 12,
        topic: "Experimentation",
        type: "multiple" as const,
        text: "Select TWO common metrics used to evaluate the quality of text generated by Large Language Models.",
        options: ["Perplexity", "Mean Squared Error (MSE)", "ROUGE Score", "Confusion Matrix"],
        correct: [0, 2],
        explanation: "Perplexity measures prediction quality (lower is better). ROUGE compares generated summaries against reference summaries."
    },
    {
        id: 13,
        topic: "Experimentation",
        type: "single" as const,
        text: "When tuning hyperparameters, what is the risk of setting the 'Learning Rate' too high?",
        options: ["The model will train too slowly", "The model may overshoot the optimal solution and fail to converge", "The model will definitely overfit", "The dataset size will increase"],
        correct: [1],
        explanation: "A high Learning Rate causes drastic weight updates, potentially overshooting the loss function minimum and causing divergence."
    },
    {
        id: 14,
        topic: "Experimentation",
        type: "single" as const,
        text: "What is 'Data Leakage' in the context of training and testing an AI model?",
        options: ["Losing data due to hard drive failure", "When information from the test set accidentally enters the training process", "Compressing data to save space", "Sharing data with unauthorized users"],
        correct: [1],
        explanation: "Data leakage happens when training data contains information that should only be in the test set, leading to overly optimistic performance metrics."
    },
    {
        id: 15,
        topic: "Experimentation",
        type: "single" as const,
        text: "Which method involves comparing two versions of a model (A and B) in a live environment to determine which performs better?",
        options: ["Backpropagation", "A/B Testing", "Cross-Validation", "Data Augmentation"],
        correct: [1],
        explanation: "A/B Testing compares two variants in a live environment to see which yields better user engagement or accuracy."
    },
    {
        id: 16,
        topic: "Data Analysis",
        type: "multiple" as const,
        text: "Which TWO steps are essential parts of the Data Preprocessing phase before training a model?",
        options: ["Normalization / Scaling", "Deploying the API", "Handling Missing Values", "Writing the Final Report"],
        correct: [0, 2],
        explanation: "Preprocessing includes Normalization (scaling data) and Handling Missing Values to ensure the model learns effectively."
    },
    {
        id: 17,
        topic: "Data Analysis",
        type: "single" as const,
        text: "Which library is built on top of Matplotlib and provides a high-level interface for drawing attractive statistical graphics?",
        options: ["TensorFlow", "Seaborn", "Scikit-learn", "OpenCV"],
        correct: [1],
        explanation: "Seaborn is based on Matplotlib and provides a high-level interface for creating attractive statistical graphics like heatmaps."
    },
    {
        id: 18,
        topic: "Data Analysis",
        type: "single" as const,
        text: "Why is it important to visualize the distribution of the target variable in a training dataset?",
        options: ["To make the dashboard look pretty", "To identify class imbalance which could bias the model", "To reduce the file size", "To increase the training speed"],
        correct: [1],
        explanation: "Visualizing the target variable helps identify class imbalance, which can bias the model toward the majority class."
    },
    {
        id: 19,
        topic: "Trustworthy AI",
        type: "multiple" as const,
        text: "Select TWO key components of Trustworthy AI.",
        options: ["Explainability / Interpretability", "Maximum Profitability", "Fairness / Bias Mitigation", "Hidden Algorithms"],
        correct: [0, 2],
        explanation: "Trustworthy AI focuses on Explainability (understanding decisions) and Fairness (preventing discrimination)."
    },
    {
        id: 20,
        topic: "Trustworthy AI",
        type: "single" as const,
        text: "What is 'Hallucination' in the context of Generative AI?",
        options: ["The model viewing images", "The model generating confident but factually incorrect information", "The model crashing due to memory errors", "The model refusing to answer a question"],
        correct: [1],
        explanation: "Hallucination is when an LLM generates fluent and confident text that is factually incorrect or nonsensical."
    }
];

// Batch 2 Data
const batch2Questions = [
    {
        id: 1,
        topic: "Core Machine Learning",
        type: "single" as const,
        text: "What is 'Transfer Learning' in the context of Large Language Models?",
        options: ["Training a model from scratch", "Using a pre-trained model and fine-tuning it on a specific task", "Transferring data between GPUs", "Converting model formats"],
        correct: [1],
        explanation: "Transfer Learning uses a model trained on massive data and fine-tunes it on a smaller, specific dataset, saving resources."
    },
    {
        id: 2,
        topic: "Core Machine Learning",
        type: "single" as const,
        text: "In a Transformer architecture, what is the purpose of 'Positional Encoding'?",
        options: ["To encrypt the data", "To inject information about the order of tokens in the sequence", "To normalize pixel values", "To reduce vocabulary size"],
        correct: [1],
        explanation: "Transformers process tokens in parallel; Positional Encoding gives the model information about the sequence order."
    },
    {
        id: 3,
        topic: "Core Machine Learning",
        type: "multiple" as const,
        text: "Select the TWO primary components of a Retrieval-Augmented Generation (RAG) system.",
        options: ["A Retriever that finds relevant documents", "A Generator (LLM) that creates the answer", "A Discriminator", "A Clustering Algorithm"],
        correct: [0, 1],
        explanation: "RAG combines a Retriever (searches external data) and a Generator (LLM synthesizes response from data)."
    },
    {
        id: 4,
        topic: "Core Machine Learning",
        type: "single" as const,
        text: "What is the primary function of the 'Softmax' activation function in the final layer of a classification network?",
        options: ["Set negative values to zero", "Convert logits into probabilities that sum to 1", "Reduce feature map dimensionality", "Prevent vanishing gradient"],
        correct: [1],
        explanation: "Softmax squashes raw logits into a probability distribution summing to 1 for multi-class classification."
    },
    {
        id: 5,
        topic: "Core Machine Learning",
        type: "single" as const,
        text: "Which technique allows fine-tuning LLMs by updating only a small subset of parameters (low-rank matrices)?",
        options: ["Dropout", "LoRA (Low-Rank Adaptation)", "Batch Normalization", "Data Augmentation"],
        correct: [1],
        explanation: "LoRA injects trainable rank decomposition matrices, significantly reducing the parameters needed for fine-tuning."
    },
    {
        id: 6,
        topic: "Core Machine Learning",
        type: "single" as const,
        text: "What is the 'Vanishing Gradient' problem in deep neural networks?",
        options: ["Model deletes data", "Gradients become so small that weights stop updating", "Learning rate is too high", "Loss becomes zero immediately"],
        correct: [1],
        explanation: "Gradients become exponentially small during backpropagation, preventing early layers from learning."
    },
    {
        id: 7,
        topic: "Software Development",
        type: "single" as const,
        text: "What is CUDA (Compute Unified Device Architecture)?",
        options: ["Python web library", "A parallel computing platform and programming model created by NVIDIA", "Database system", "Neural network architecture"],
        correct: [1],
        explanation: "CUDA allows developers to use NVIDIA GPUs for general purpose processing (GPGPU), accelerating AI tasks."
    },
    {
        id: 8,
        topic: "Software Development",
        type: "single" as const,
        text: "Which file format is designed as an open standard for representing machine learning models (interoperability)?",
        options: ["JSON", "ONNX (Open Neural Network Exchange)", "CSV", "HTML"],
        correct: [1],
        explanation: "ONNX allows models to be trained in one framework and deployed in another."
    },
    {
        id: 9,
        topic: "Software Development",
        type: "multiple" as const,
        text: "Select TWO benefits of Microservices architecture for AI compared to Monolithic.",
        options: ["Independent scaling of components", "Tighter coupling", "Technology diversity", "Easier debugging for beginners"],
        correct: [0, 2],
        explanation: "Microservices allow independent scaling and the use of different technology stacks for different components."
    },
    {
        id: 10,
        topic: "Software Development",
        type: "single" as const,
        text: "What is the primary purpose of 'pip freeze > requirements.txt'?",
        options: ["Freeze code editing", "List all installed packages and versions for reproducibility", "Uninstall packages", "Cool down GPU"],
        correct: [1],
        explanation: "It generates a list of installed packages and versions to recreate the environment elsewhere."
    },
    {
        id: 11,
        topic: "Software Development",
        type: "single" as const,
        text: "Which NVIDIA tool serves deep learning models from any framework for high-performance inference?",
        options: ["GeForce Experience", "NVIDIA Triton Inference Server", "NVIDIA Canvas", "NVIDIA Broadcast"],
        correct: [1],
        explanation: "Triton Inference Server optimizes and serves AI models from major frameworks on GPUs/CPUs."
    },
    {
        id: 12,
        topic: "Experimentation",
        type: "single" as const,
        text: "What is the effect of setting 'Temperature' to 0 in LLM generation?",
        options: ["Output becomes random", "Output becomes deterministic and repetitive", "Stops generating", "Runs faster"],
        correct: [1],
        explanation: "Temperature 0 makes the model deterministic, always choosing the most probable next token."
    },
    {
        id: 13,
        topic: "Experimentation",
        type: "single" as const,
        text: "What is the purpose of a 'Validation Set'?",
        options: ["Update weights", "Final evaluation", "Tune hyperparameters and prevent overfitting during training", "Store rejected data"],
        correct: [2],
        explanation: "The Validation Set evaluates the model during training to tune parameters and detect overfitting."
    },
    {
        id: 14,
        topic: "Experimentation",
        type: "multiple" as const,
        text: "When benchmarking AI models, which TWO metrics are most distinct and commonly measured?",
        options: ["Latency", "Throughput", "Electricity Bill", "Fan Speed"],
        correct: [0, 1],
        explanation: "Latency is response time for one request; Throughput is volume of requests per unit of time."
    },
    {
        id: 15,
        topic: "Experimentation",
        type: "single" as const,
        text: "What is 'K-Fold Cross-Validation'?",
        options: ["Training K times on same data", "Splitting data into K subsets, training on K-1, validating on 1, repeating K times", "Validating with K algorithms", "Removing K rows"],
        correct: [1],
        explanation: "It ensures every data point is used for both training and validation to assess robustness."
    },
    {
        id: 16,
        topic: "Data Analysis",
        type: "single" as const,
        text: "Which technique visualizes high-dimensional data in 2D/3D preserving local structure?",
        options: ["Linear Regression", "t-SNE", "Random Forest", "SQL"],
        correct: [1],
        explanation: "t-SNE maps high-dimensional data to 2D/3D, revealing structures and clusters."
    },
    {
        id: 17,
        topic: "Data Analysis",
        type: "multiple" as const,
        text: "Select TWO visualization types best for identifying relationships or correlations.",
        options: ["Scatter Plot", "Pie Chart", "Heatmap (Correlation Matrix)", "Gauge Chart"],
        correct: [0, 2],
        explanation: "Scatter plots show relationships between two variables; Heatmaps show correlation strength."
    },
    {
        id: 18,
        topic: "Data Analysis",
        type: "single" as const,
        text: "Which plot is best for detecting outliers via distribution quartiles?",
        options: ["Bar Chart", "Box Plot", "Line Chart", "Donut Chart"],
        correct: [1],
        explanation: "Box Plots display quartiles; points outside the 'whiskers' are potential outliers."
    },
    {
        id: 19,
        topic: "Trustworthy AI",
        type: "multiple" as const,
        text: "Which TWO scenarios are security risks specific to Generative AI?",
        options: ["Prompt Injection", "SQL Injection", "Model Inversion / Data Extraction", "Server Failure"],
        correct: [0, 2],
        explanation: "Prompt Injection bypasses filters; Model Inversion reconstructs training data."
    },
    {
        id: 20,
        topic: "Trustworthy AI",
        type: "single" as const,
        text: "What is the purpose of a 'Model Card'?",
        options: ["Hardware card", "Document providing model details, limits, and metrics", "License key", "Billing card"],
        correct: [1],
        explanation: "Model Cards provide transparency regarding intended use, limitations, and performance."
    }
];

// Batch 3 Data
const batch3Questions = [
    {
        id: 1,
        topic: "Core Machine Learning",
        type: "single" as const,
        text: "What is 'RLHF' primarily used for in training Modern LLMs?",
        options: ["Compress model size", "Align outputs with human intent and safety", "Increase vocabulary", "Generate images"],
        correct: [1],
        explanation: "RLHF optimizes the model using human feedback to ensure safety, helpfulness, and alignment."
    },
    {
        id: 2,
        topic: "Core Machine Learning",
        type: "single" as const,
        text: "Which model type adds noise to data then learns to reverse it to generate new data?",
        options: ["Transformer", "Diffusion Models", "Linear Regression", "K-Means"],
        correct: [1],
        explanation: "Diffusion Models generate data by learning to reverse a noise-adding process."
    },
    {
        id: 3,
        topic: "Core Machine Learning",
        type: "multiple" as const,
        text: "Select TWO common Activation Functions for hidden layers.",
        options: ["ReLU", "Sigmoid", "Linear", "Step"],
        correct: [0, 1],
        explanation: "ReLU is efficient and common; Sigmoid is a classic non-linear function."
    },
    {
        id: 4,
        topic: "Core Machine Learning",
        type: "single" as const,
        text: "What does 'High Bias' typically indicate in the Bias-Variance tradeoff?",
        options: ["Overfitting", "Underfitting", "Perfect model", "Slow training"],
        correct: [1],
        explanation: "High Bias indicates underfitting; the model is too simple to capture patterns."
    },
    {
        id: 5,
        topic: "Core Machine Learning",
        type: "single" as const,
        text: "Which Loss Function is standard for Multi-Class Classification?",
        options: ["MSE", "Categorical Cross-Entropy", "Hinge Loss", "Absolute Error"],
        correct: [1],
        explanation: "Categorical Cross-Entropy measures the difference between predicted and actual probability distributions."
    },
    {
        id: 6,
        topic: "Core Machine Learning",
        type: "single" as const,
        text: "What is 'Zero-Shot Learning'?",
        options: ["Training with zero data", "Performing tasks without specific training examples", "Zero latency", "Resetting weights"],
        correct: [1],
        explanation: "Zero-Shot Learning is handling tasks/categories not explicitly seen during training."
    },
    {
        id: 7,
        topic: "Software Development",
        type: "single" as const,
        text: "Which NVIDIA SDK optimizes deep learning models for inference via layer fusion?",
        options: ["NVIDIA TensorRT", "CUDA Toolkit", "Omniverse", "Modulus"],
        correct: [0],
        explanation: "TensorRT optimizes trained models for high-performance inference on GPUs."
    },
    {
        id: 8,
        topic: "Software Development",
        type: "single" as const,
        text: "What is the purpose of 'Virtual Environments' in Python?",
        options: ["Speed up internet", "Isolate project dependencies", "Increase RAM", "Run in browser"],
        correct: [1],
        explanation: "Virtual Environments isolate dependencies to prevent conflicts between projects."
    },
    {
        id: 9,
        topic: "Software Development",
        type: "multiple" as const,
        text: "Select TWO standard stages in a CI/CD pipeline for MLOps.",
        options: ["Automated Model Testing", "Manual Soldering", "Model Registry / Deployment", "Data Deletion"],
        correct: [0, 2],
        explanation: "MLOps pipelines automate testing, versioning (registry), and deployment."
    },
    {
        id: 10,
        topic: "Software Development",
        type: "single" as const,
        text: "Which columnar file format is preferred for Big Data storage?",
        options: ["TXT", "Parquet", "XML", "BMP"],
        correct: [1],
        explanation: "Parquet is a compressed, columnar format efficient for big data queries."
    },
    {
        id: 11,
        topic: "Software Development",
        type: "single" as const,
        text: "Which data-interchange format is commonly used for AI API payloads?",
        options: ["JSON", "Assembly", "PDF", "MP3"],
        correct: [0],
        explanation: "JSON is the standard, human-readable format for REST APIs."
    },
    {
        id: 12,
        topic: "Experimentation",
        type: "single" as const,
        text: "Which metric minimizes False Positives in classification?",
        options: ["Recall", "Precision", "Accuracy", "MAE"],
        correct: [1],
        explanation: "Precision measures accuracy of positive predictions; high precision means fewer False Positives."
    },
    {
        id: 13,
        topic: "Experimentation",
        type: "single" as const,
        text: "What is 'Top-p' (Nucleus) sampling?",
        options: ["Top 10 words", "Smallest set of tokens with cumulative probability > p", "Top of hard drive", "Longest words"],
        correct: [1],
        explanation: "Top-p selects from a dynamic set of high-probability tokens, improving naturalness."
    },
    {
        id: 14,
        topic: "Experimentation",
        type: "single" as const,
        text: "What is 'Early Stopping'?",
        options: ["Stopping immediately", "Halting training when validation performance plateaus", "Stopping when hot", "Stopping at zero loss"],
        correct: [1],
        explanation: "Early Stopping prevents overfitting by halting training when validation loss stops improving."
    },
    {
        id: 15,
        topic: "Experimentation",
        type: "multiple" as const,
        text: "Select TWO indicators of Overfitting.",
        options: ["Low Training Loss", "High Validation Loss", "Both high", "Small model"],
        correct: [0, 1],
        explanation: "Overfitting is the gap between low training loss (memorization) and high validation loss (poor generalization)."
    },
    {
        id: 16,
        topic: "Data Analysis",
        type: "single" as const,
        text: "What is 'One-Hot Encoding' used for?",
        options: ["Compressing video", "Converting categories to binary vectors", "Encryption", "Sorting"],
        correct: [1],
        explanation: "One-Hot Encoding converts categorical data into numerical binary vectors."
    },
    {
        id: 17,
        topic: "Data Analysis",
        type: "single" as const,
        text: "Which visualization shows frequency distribution of a numerical variable?",
        options: ["Scatter Plot", "Histogram", "Network Graph", "Word Cloud"],
        correct: [1],
        explanation: "Histograms display the frequency distribution of continuous data using bins."
    },
    {
        id: 18,
        topic: "Data Analysis",
        type: "multiple" as const,
        text: "Which TWO Pandas functions are used for initial exploration?",
        options: ["df.head()", "df.delete_all()", "df.describe()", "df.format()"],
        correct: [0, 2],
        explanation: "df.head() shows first rows; df.describe() shows summary statistics."
    },
    {
        id: 19,
        topic: "Trustworthy AI",
        type: "single" as const,
        text: "What is an 'Adversarial Attack'?",
        options: ["Stealing server", "Subtle input perturbations to trick the model", "Negative review", "Overloading API"],
        correct: [1],
        explanation: "Adversarial attacks use imperceptible noise to cause model misclassification."
    },
    {
        id: 20,
        topic: "Trustworthy AI",
        type: "multiple" as const,
        text: "Select TWO methods for data privacy protection.",
        options: ["Anonymization", "Federated Learning", "Publishing raw data", "Plain text passwords"],
        correct: [0, 1],
        explanation: "Anonymization removes PII; Federated Learning trains on decentralized data."
    }
];

// Batch 4 Data
const batch4Questions = [
    {
        id: 1,
        topic: "Core Machine Learning",
        type: "single" as const,
        text: "Which describes the BERT architecture?",
        options: ["Decoder-only", "Encoder-only", "Encoder-Decoder", "RNN"],
        correct: [1],
        explanation: "BERT is Encoder-only, designed to understand context bidirectionally."
    },
    {
        id: 2,
        topic: "Core Machine Learning",
        type: "single" as const,
        text: "What does the 'Context Window' limit refer to?",
        options: ["Training speed", "Max tokens in input+output", "Languages supported", "Disk size"],
        correct: [1],
        explanation: "Context Window is the maximum sequence length the model can process at once."
    },
    {
        id: 3,
        topic: "Core Machine Learning",
        type: "multiple" as const,
        text: "Select TWO benefits of 'Chain-of-Thought' (CoT) prompting.",
        options: ["Reduces cost", "Breaks down reasoning steps", "Helps complex logic problems", "Converts to database"],
        correct: [1, 2],
        explanation: "CoT encourages step-by-step reasoning, improving performance on complex tasks."
    },
    {
        id: 4,
        topic: "Core Machine Learning",
        type: "single" as const,
        text: "What is a 'Multimodal' model?",
        options: ["Runs on multiple PCs", "Processes multiple media types (text, image, etc.)", "Multiple learning rates", "Multiple outputs"],
        correct: [1],
        explanation: "Multimodal models understand relationships between different data types (e.g., text and images)."
    },
    {
        id: 5,
        topic: "Core Machine Learning",
        type: "single" as const,
        text: "What is 'Latent Space'?",
        options: ["Generation time", "Compressed representation where similar data is close", "GPU memory space", "Server room"],
        correct: [1],
        explanation: "Latent Space is a compressed representation where semantically similar items are geometrically close."
    },
    {
        id: 6,
        topic: "Core Machine Learning",
        type: "single" as const,
        text: "What is the role of 'Stochastic Gradient Descent' (SGD)?",
        options: ["Initialize weights", "Iteratively update weights using mini-batches", "Shuffle dataset", "Increase learning rate"],
        correct: [1],
        explanation: "SGD updates weights based on random mini-batches to minimize loss efficiently."
    },
    {
        id: 7,
        topic: "Software Development",
        type: "single" as const,
        text: "What is NVIDIA NeMo used for?",
        options: ["Video games", "Building/training Generative AI models", "SQL queries", "UI design"],
        correct: [1],
        explanation: "NeMo is a framework for building and training LLMs, Speech AI, and Multimodal models."
    },
    {
        id: 8,
        topic: "Software Development",
        type: "single" as const,
        text: "What does HTTP 429 indicate?",
        options: ["Server Error", "Not Found", "Too Many Requests (Rate Limit)", "Unauthorized"],
        correct: [2],
        explanation: "HTTP 429 means the client exceeded the API rate limit."
    },
    {
        id: 9,
        topic: "Software Development",
        type: "multiple" as const,
        text: "Select TWO differences between Docker and VMs.",
        options: ["Containers share host kernel", "VMs are lighter", "VMs include full OS", "Docker can't run on Linux"],
        correct: [0, 2],
        explanation: "Containers share the kernel (lightweight); VMs emulate full hardware/OS (heavy)."
    },
    {
        id: 10,
        topic: "Software Development",
        type: "single" as const,
        text: "Which format is used for AI config files (human-readable, indentation)?",
        options: ["Binary", "YAML", "HTML", "C++"],
        correct: [1],
        explanation: "YAML is human-readable and standard for configurations."
    },
    {
        id: 11,
        topic: "Software Development",
        type: "single" as const,
        text: "What does 'pip install -r requirements.txt' do?",
        options: ["Uninstall Python", "Install dependencies listed in the file", "Create text file", "Run script"],
        correct: [1],
        explanation: "It installs all specified libraries and versions from the file."
    },
    {
        id: 12,
        topic: "Experimentation",
        type: "single" as const,
        text: "What does BLEU score evaluate?",
        options: ["GPU speed", "Machine translation quality vs human reference", "Image resolution", "Sentiment"],
        correct: [1],
        explanation: "BLEU measures n-gram overlap between machine output and human reference translations."
    },
    {
        id: 13,
        topic: "Experimentation",
        type: "multiple" as const,
        text: "Select TWO Hyperparameter Tuning strategies.",
        options: ["Grid Search", "Manual Guessing", "Random Search", "Deleting Dataset"],
        correct: [0, 2],
        explanation: "Grid Search (exhaustive) and Random Search (efficient) are standard strategies."
    },
    {
        id: 14,
        topic: "Experimentation",
        type: "single" as const,
        text: "What is a 'False Negative'?",
        options: ["Correct negative", "Incorrect positive", "Incorrectly predicted negative (missed positive)", "Crash"],
        correct: [2],
        explanation: "False Negative is failing to detect a positive case (predicting negative when it is positive)."
    },
    {
        id: 15,
        topic: "Experimentation",
        type: "single" as const,
        text: "If training loss decreases but validation loss increases, what is happening?",
        options: ["Underfitting", "Overfitting", "Convergence", "Low learning rate"],
        correct: [1],
        explanation: "Divergence between training and validation loss indicates overfitting."
    },
    {
        id: 16,
        topic: "Data Analysis",
        type: "multiple" as const,
        text: "Select TWO techniques for Missing Data.",
        options: ["Imputation", "Deleting rows", "Ignoring", "Multiplying by zero"],
        correct: [0, 1],
        explanation: "Imputation (filling) and Deletion are standard handling methods."
    },
    {
        id: 17,
        topic: "Data Analysis",
        type: "single" as const,
        text: "What does Correlation Coefficient 0 imply?",
        options: ["Perfect positive", "Perfect negative", "No linear relationship", "Corruption"],
        correct: [2],
        explanation: "0 implies no linear correlation between variables."
    },
    {
        id: 18,
        topic: "Data Analysis",
        type: "single" as const,
        text: "What is 'Feature Scaling' used for?",
        options: ["Increase features", "Normalize range of variables", "Remove target", "Visualize"],
        correct: [1],
        explanation: "Scaling ensures features share a common range, crucial for distance-based algorithms."
    },
    {
        id: 19,
        topic: "Trustworthy AI",
        type: "single" as const,
        text: "What is 'Selection Bias'?",
        options: ["Selecting best architecture", "Training data not representing real-world population", "Wrong GPU", "Rate bias"],
        correct: [1],
        explanation: "Selection Bias happens when training data fails to represent the target population."
    },
    {
        id: 20,
        topic: "Trustworthy AI",
        type: "multiple" as const,
        text: "Select TWO IP risks in Generative AI.",
        options: ["Training on copyrighted data", "Memory error", "Generating infringing output", "Open source"],
        correct: [0, 2],
        explanation: "Risks include input liability (training data) and output liability (infringement)."
    }
];

export const batches: Batch[] = [
    { id: 'b1', name: 'Practice Batch 1', questions: batch1Questions },
    { id: 'b2', name: 'Practice Batch 2', questions: batch2Questions },
    { id: 'b3', name: 'Practice Batch 3', questions: batch3Questions },
    { id: 'b4', name: 'Practice Batch 4', questions: batch4Questions },
];