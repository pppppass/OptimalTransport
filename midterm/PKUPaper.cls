% !TeX encoding = UTF-8
% !TeX program = LuaLaTeX
% !TeX spellcheck = LaTeX

% Author : lzh
% Description : Template for PKU-style paper, mainly modified from pkuthss package, only for personal use

% This file cannot be typeset directly.

\ProvidesClass{PKUPaper}[2017/10/25 PKU-style paper class]

\DeclareOption{english}{\def\PKUPaper@english\relax}
\ProcessOptions

\PassOptionsToPackage{no-math}{fontspec}

\LoadClass[UTF8, hyperref, cs4size, fntef, fancyhdr, twoside]{ctexart}[2011/03/11]

\RequirePackage{ifpdf, ifxetex}
\RequirePackage{graphicx}[1999/02/16]
\RequirePackage{geometry}
\RequirePackage[defaultsups]{newtxtext}
\RequirePackage{newtxmath}
\RequirePackage{latexsym}
\RequirePackage{tikz}
\RequirePackage[cleardoublepage = current]{scrextend}
\RequirePackage[titles]{tocloft}
\RequirePackage{caption, subcaption}
\RequirePackage{setspace}
\RequirePackage{enumitem}


\newcommand*\pkuthss@int@circled[1]{%
	\scalebox{0.8}{\tikz[baseline = (char.base)]{
		\node[
			shape = circle, draw = black, minimum size = 1.25em, inner sep = 0pt
		] (char) {#1};
	}}%
}

\renewcommand\thefootnote%
	{\protect\pkuthss@int@circled{\arabic{footnote}}}

\deffootnote{2\ccwd}{3.5\ccwd}{\thefootnotemark\hspace{0.75\ccwd}}

\linespread{1.39}\selectfont

\DeclareCaptionFont{cfive}{\zihao{5}}
\DeclareCaptionLabelSeparator{quad}{\quad}

\captionsetup{font = cfive, labelsep = quad}

\setlist{nolistsep}

\renewcommand{\numberline}[1]{%
	\settowidth\@tempdimb{#1\hspace{0.5em}}%
	\ifdim\@tempdima<\@tempdimb%
			\@tempdima=\@tempdimb%
	\fi%
	\hb@xt@\@tempdima{\@cftbsnum #1\@cftasnum\hfil}\@cftasnumb%
}

\geometry{
	a4paper, includefoot, hmargin = 2.6cm, top = 3cm, bottom = 2.5cm,
	headheight = 1.5cm, headsep = 0.5cm, footskip = 0.75cm
}

\setlength{\abovedisplayskip}{6bp plus 1.5bp minus 3.5bp}
\setlength{\abovedisplayshortskip}{3bp plus 0.75bp minus 1.75bp}
\setlength{\belowdisplayshortskip}{3bp plus 0.75bp minus 1.75bp}
\setlength{\belowdisplayskip}{6bp plus 1.5bp minus 3.5bp}

\ctexset{
	section = {
		format = {\zihao {-3}\bfseries },
		beforeskip = {20bp plus 1ex minus 0.2ex},
		afterskip = {6bp plus 0.2ex}
	},
	subsection = {
		format = {\zihao {4}\bfseries },
		beforeskip = {12bp plus 1ex minus 0.2ex},
		afterskip = {6bp plus 0.2ex}
	},
	subsubsection = {
		format = {\bfseries },
		beforeskip = {12bp plus 1ex minus 0.2ex},
		afterskip = {6bp plus 0.2ex}
	}
}

\def\pkuthss@int@setcase#1{%
	\let\MakeUppercase\relax%
	\expandafter\let\csname MakeUppercase \endcsname\relax%
	#1%
}

\fancypagestyle{plain}{
	\fancyhf{}\renewcommand*{\headrulewidth}{0.75bp}
	\fancyfoot[C]{\zihao{5}\normalfont{\thepage}}
	\fancyhead[CE]{\zihao{5}\normalfont{\cuniversity \cthesisname}}
	\fancyhead[CO]{\zihao{5}\normalfont\pkuthss@int@setcase{\titlemark}}
}

\pagestyle{plain}

\ifdefined\PKUPaper@english%
\renewcommand\figurename{Figure}%
\fi
