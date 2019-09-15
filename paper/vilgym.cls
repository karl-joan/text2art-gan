% !TeX program = xelatex
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{vilgym}[2018/12/26 0.3 Viljandi Gymnasium XeLaTeX research paper class]

\DeclareOption{draft}{\PassOptionsToPackage{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[12pt, a4paper]{article}

% Require LaTeX3
\RequirePackage{expl3,xparse}

% Use Times New Roman. Requirex XeLaTeX.
\RequirePackage{fontspec}
\setmainfont{Times New Roman}

% Page margins
\RequirePackage{geometry}
\geometry{left=30mm, right=20mm, top=25mm, bottom=25mm}

% Page numbering
\RequirePackage{fancyhdr}
\pagenumbering{arabic}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\rfoot{\thepage}
\pagestyle{fancy}

% Set spacing
\RequirePackage{setspace}
\spacing{1.4} % Roughly equivalent to Microsoft Word's 1.5 spacing.

% For good automatic hyphenation. Disable use of tilde for accents.
\RequirePackage[estonian .notilde]{babel}

%% Disable shorthands
\let\LanguageShortHands\languageshorthands
\def\languageshorthands#1{}

% Use Estonian quotation marks.
\usepackage[autostyle]{csquotes}
\DeclareQuoteAlias{dutch}{estonian}

% Start each section on a new page
\newcommand{\sectionbreak}{\newpage}

% Pragraph skip and indent
\setlength{\parskip}{12pt}
\setlength{\parindent}{0pt}

% Tweak skip for different list environments
\RequirePackage{enumitem}
\setlist[itemize]{topsep=0pt, itemsep=0pt}
\setlist[enumerate]{topsep=0pt, itemsep=0pt}
\setlist[description]{topsep=0pt, itemsep=0pt}

% Fancy linking
\RequirePackage[hidelinks]{hyperref}
\urlstyle{same} % Use the main font

% Make URL line breaking much less strict
\expandafter\def\expandafter\UrlBreaks\expandafter{\UrlBreaks
	\do\a\do\b\do\c\do\d\do\e\do\f\do\g\do\h\do\i\do\j%
	\do\k\do\l\do\m\do\n\do\o\do\p\do\q\do\r\do\s\do\t%
	\do\u\do\v\do\w\do\x\do\y\do\z\do\A\do\B\do\C\do\D%
	\do\E\do\F\do\G\do\H\do\I\do\J\do\K\do\L\do\M\do\N%
	\do\O\do\P\do\Q\do\R\do\S\do\T\do\U\do\V\do\W\do\X%
	\do\Y\do\Z\do\1\do\2\do\3\do\4\do\5\do\6\do\7\do\8%
	\do\9}

% Make description labels linkable
\let\originaldescriptionlabel\descriptionlabel
\renewcommand*{\descriptionlabel}[1]{\phantomsection\originaldescriptionlabel{#1}}

% Unnumbered sections
\newcommand*{\unsection}[1]{\newpage\phantomsection\section*{#1}\addcontentsline{toc}{section}{#1}}
\newcommand*{\unsubsection}[1]{\phantomsection\subsection*{#1}\addcontentsline{toc}{subsection}{#1}}

% Modify caption label format
\RequirePackage{caption}
\DeclareCaptionFormat{vilgymcaption}{\fontsize{10pt}{\f@baselineskip}\selectfont#1#2#3}
\captionsetup{justification=raggedright,singlelinecheck=false}
\captionsetup[figure]{name={Joonis},labelsep=period,format=vilgymcaption}
\captionsetup[table]{name={Tabel},labelsep=period,format=vilgymcaption}

%% Footnotes
% Make sure footnotes are properly left-aligned and at the bottom of the page
\RequirePackage[hang,flushmargin,bottom]{footmisc}
\renewcommand*{\footnotelayout}{\fontsize{10pt}{\f@baselineskip}\selectfont} % Seems to be 10pt by default, but make sure.

% Add footnote contents to the bibliography
\ExplSyntaxOn

\seq_new:N \l_vilgym_footnote_seq % Sequence of bibliography entries from footnotes.
\int_new:N \l_vilgym_footnote_num % Footnote number counter. Used for bibliography labels.
\tl_new:N \l_vilgym_footnote_tl % Footnote bibliography label.

\let\originalfootnote\footnote
\RenewDocumentCommand\footnote{m}
{
	\originalfootnote{#1}
	\tl_set:Nx \l_vilgym_footnote_tl { { \int_use:N \l_vilgym_footnote_num }~ }
	\seq_put_right:NV \l_vilgym_footnote_seq {\l_vilgym_footnote_tl #1}
	\int_add:Nn \l_vilgym_footnote_num {1}
}

%%% Bibliography related stuff
%% Change the the look of the bibliography and citation
\RequirePackage[citestyle=authoryear,maxbibnames=99]{biblatex}
\DefineBibliographyStrings{estonian}{andothers={jt\adddot}} % Replace et al with jt.
\renewcommand*{\nameyeardelim}{\addcomma\space} % Add a comma between author and year

% Removes the numbers before entries
\DeclareFieldFormat{labelnumberwidth}{} % Doesn't print anything in the label
\setlength{\biblabelsep}{0pt} % Eliminates the spacing before the entries

% Title formatting
\RequirePackage{titlesec}
\titleformat{\subsection}[hang]{\bfseries}{\thesubsection}{12pt}{}
\titleformat{\section}[hang]{\phantomsection\bfseries}{\thesection}{12pt}{\fontsize{16pt}{\f@baselineskip}\selectfont\MakeUppercase}

\let\originaltableofcontents\tableofcontents
\RenewDocumentCommand{\tableofcontents}{}
{
	\titleformat{\section}{\bfseries}{\thesection}{12pt}{\fontsize{16pt}{\f@baselineskip}\selectfont\MakeUppercase}
	\originaltableofcontents
}

% TOC formatting
\RequirePackage{titletoc}
\dottedcontents{subsection}[40pt]{}{25pt}{4pt}
\dottedcontents{subsubsection}[75pt]{}{35pt}{4pt}
\titlecontents{section}[1pt]
{}
{\hspace*{15pt}\contentslabel{15pt}\uppercase}
{\uppercase}
{\titlerule*[4pt]{.}\contentspage}

%% For appendices
% The appendices section. Changes the TOC formatting for any subsequent subsections (there should be no other subsections than appendices after this).
\NewDocumentCommand{\appendices}{}
{
	% Redefine the subsection TOC formatting, so appendixes don't have page numbers.
	% TODO: the dots don't extend to the right edge
	\titlecontents{subsection}[15pt]
	{}
	{\hspace*{15pt}\contentslabel{15pt}\uppercase}
	{}
	{}

	\unsection{Lisad}
}

% For creating and automatically labelling an appendix subsection.
\int_new:N \l_vilgym_appendix_num
\int_set:Nn \l_vilgym_appendix_num { 1 }

\RenewDocumentCommand{\appendix}{m}
{
	\unsubsection{Lisa~\int_use:N \l_vilgym_appendix_num.~#1}
	\label{appendix:\int_use:N \l_vilgym_appendix_num}

	\int_add:Nn \l_vilgym_appendix_num {1}
}

% A shorthand for linking to an appendix.
\NewDocumentCommand{\seeappendix}{m}
{
	(\hyperref[appendix:#1]{vt~lisa~#1})
}
\ExplSyntaxOff

%% Title page
% TODO: Update to LaTeX3.
% Some setup for titlepage configuration
\newtoks\type
\newtoks\authortext
\newtoks\instructortext
\newcommand*{\instructor}[1]{\instructortext{Juhendaja #1}}
\newcommand*{\instructors}[1]{\instructortext{Juhendajad #1}}
\renewcommand*{\author}[1]{\authortext{Autor #1}}
\newcommand*{\authors}[1]{\authortext{Autorid #1}}

\renewcommand*{\maketitle}{%
	\begin{titlepage}
		\centering
		\fontsize{16pt}{\f@baselineskip}\selectfont
		Viljandi Gümnaasium
		
		\vfill
		
		\fontsize{20pt}{\f@baselineskip}\selectfont
		\textbf{\@title}
		
		\fontsize{16pt}{\f@baselineskip}\selectfont
		\edef\1{\the\type}
		\ifx\1\empty
			\type{Uurimistöö}
		\fi
		
		\the\type
		
		\begin{flushright}
			\singlespacing
			\fontsize{16pt}{\f@baselineskip}\selectfont
			\the\authortext

			\the\instructortext
		\end{flushright}
		
		\vfill
		
		\fontsize{16pt}{\f@baselineskip}\selectfont
		Viljandi \@date
	\end{titlepage}
	
	% Titlepage keeps the page counter at 1, so we need to manually set it to 2.
	\setcounter{page}{2}%
}
