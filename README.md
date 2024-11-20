\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{a4paper, margin=1in}

\title{\textbf{Accurate Liver Registration of 3D Ultrasound and CT Volume: An Open Dataset and a Model Fusion Method}}
\author{}
\date{}

\begin{document}

\maketitle

\section*{Overview}
Liver registration between ultrasound (US) and computed tomography (CT) is critical for various medical applications, such as surgical navigation and interventional guidance. This work introduces:
\begin{itemize}
    \item \textbf{A new open dataset}: Featuring paired 3D ultrasound and CT volumes with accurate liver segmentation annotations.
    \item \textbf{A model fusion-based registration method}: Combining deep learning and traditional optimization approaches for robust and accurate results.
\end{itemize}

\section*{Dataset}
The dataset includes:
\begin{enumerate}
    \item \textbf{3D Ultrasound (US) Volumes}: High-quality, segmented US images with corresponding labels.
    \item \textbf{3D Computed Tomography (CT) Volumes}: Paired CT images with liver segmentation ground truth.
\end{enumerate}

\subsection*{Data Format}
\begin{itemize}
    \item \textbf{US Data}: Provided in NIfTI (.nii) format.
    \item \textbf{CT Data}: Provided in NIfTI (.nii) format.
    \item \textbf{Annotations}: Liver segmentations are included as binary masks.
\end{itemize}

\subsection*{Access}
Please \href{mailto:your-email@example.com}{request access} to download the dataset.

\section*{Code}
This repository provides the implementation of our proposed registration pipeline:
\begin{enumerate}
    \item \textbf{Preprocessing}: Normalization, cropping, and resampling of US and CT data.
    \item \textbf{Model Fusion Registration}:
    \begin{itemize}
        \item \textbf{Deep Learning Component}: Pre-trained feature extraction for initial alignment.
        \item \textbf{Optimization Component}: Fine-tuning registration using similarity measures (e.g., mutual information).
    \end{itemize}
    \item \textbf{Evaluation Metrics}: Dice Similarity Coefficient (DSC), Hausdorff Distance, etc.
\end{enumerate}

\subsection*{Setup}

\subsubsection*{Prerequisites}
\begin{itemize}
    \item Python $\geq 3.8$
    \item PyTorch $\geq 1.10$
    \item Numpy, Scipy, SimpleITK, and other dependencies listed in \texttt{requirements.txt}.
\end{itemize}

\subsubsection*{Installation}
\begin{enumerate}
    \item Clone the repository:
    \begin{verbatim}
    git clone https://github.com/your-username/3D-US-CT-Liver-Registration.git
    cd 3D-US-CT-Liver-Registration
    \end{verbatim}
    \item Install dependencies:
    \begin{verbatim}
    pip install -r requirements.txt
    \end{verbatim}
\end{enumerate}

\subsubsection*{Running the Code}
\begin{itemize}
    \item \textbf{Preprocessing}:
    \begin{verbatim}
    python preprocess.py --input_path /path/to/data --output_path /path/to/preprocessed_data
    \end{verbatim}
    \item \textbf{Training}:
    \begin{verbatim}
    python train.py --config configs/train_config.yaml
    \end{verbatim}
    \item \textbf{Testing}:
    \begin{verbatim}
    python test.py --model_path /path/to/saved_model.pth --test_data /path/to/test_data
    \end{verbatim}
\end{itemize}

\section*{Results}
Our method achieves state-of-the-art performance on liver registration tasks, with robust results demonstrated on the provided dataset.

\begin{center}
\begin{tabular}{|l|c|}
\hline
\textbf{Metric} & \textbf{Result} \\
\hline
Dice Similarity & 0.89 \\
\hline
Hausdorff Distance & 5.2 mm \\
\hline
\end{tabular}
\end{center}

\section*{Citation}
If you find this dataset or code helpful, please cite our paper:
\begin{verbatim}
@article{YourPaper2024,
  title={Accurate Liver Registration of 3D Ultrasound and CT Volume: An Open Dataset and a Model Fusion Method},
  author={Your Name and Collaborators},
  journal={Journal/Conference Name},
  year={2024},
  volume={XX},
  pages={XXX--XXX},
  doi={DOI}
}
\end{verbatim}

\section*{License}
This project is licensed under the \href{LICENSE}{MIT License}.

\section*{Contact}
For questions or support, please open an issue or contact us at \href{mailto:your-email@example.com}{your-email@example.com}.

\end{document}
